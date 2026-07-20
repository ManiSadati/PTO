#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void LaunchSoftmaxKernel(void *data, void *cast_1, void *stream);

#define CHECK(x)                                                               \
  do {                                                                         \
    auto e = (x);                                                              \
    if ((int)e) {                                                              \
      std::printf("%s failed %d\n", #x, (int)e);                              \
      return 1;                                                                \
    }                                                                          \
  } while (0)

namespace {

constexpr int kBlocks = 8;
constexpr int kSoftmaxGroupsPerBlock = 2;
constexpr int kSoftmaxLength = 1024;
constexpr int kElementsPerBlock =
    kSoftmaxGroupsPerBlock * kSoftmaxLength;
constexpr int kTotalElements = kBlocks * kElementsPerBlock;

bool ReadBinaryFile(const char *path, void *dst, size_t expectedBytes) {
  FILE *f = std::fopen(path, "rb");
  if (!f) {
    std::perror(path);
    return false;
  }
  if (std::fseek(f, 0, SEEK_END) != 0) {
    std::perror("fseek");
    std::fclose(f);
    return false;
  }
  const long fileSize = std::ftell(f);
  if (fileSize < 0 || static_cast<size_t>(fileSize) != expectedBytes) {
    std::printf("File size mismatch for %s: expected %zu bytes, got %ld bytes\n",
                path, expectedBytes, fileSize);
    std::fclose(f);
    return false;
  }
  std::rewind(f);
  const size_t nread = std::fread(dst, 1, expectedBytes, f);
  std::fclose(f);
  if (nread != expectedBytes) {
    std::printf("Short read for %s: expected %zu bytes, got %zu bytes\n", path,
                expectedBytes, nread);
    return false;
  }
  return true;
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h >> 15);
  const uint32_t exp = static_cast<uint32_t>((h >> 10) & 0x1f);
  uint32_t mant = static_cast<uint32_t>(h & 0x03ff);

  uint32_t out = 0;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      int e = -14;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --e;
      }
      mant &= 0x03ff;
      out = (sign << 31) | (static_cast<uint32_t>(e + 127) << 23) |
            (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    out = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
  }

  union {
    uint32_t u;
    float f;
  } v{out};
  return v.f;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::printf("Usage: %s <input_fp16.bin> <golden_fp16.bin>\n", argv[0]);
    return 1;
  }

  std::vector<uint16_t> input(kTotalElements);
  std::vector<uint16_t> golden(kTotalElements);
  const size_t bytes = input.size() * sizeof(uint16_t);
  if (!ReadBinaryFile(argv[1], input.data(), bytes) ||
      !ReadBinaryFile(argv[2], golden.data(), bytes)) {
    return 1;
  }

  aclInit(nullptr);
  CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  CHECK(aclrtCreateStream(&stream));

  void *gmIn = nullptr;
  void *gmOut = nullptr;
  CHECK(aclrtMalloc(&gmIn, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&gmOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMemcpy(gmIn, bytes, input.data(), bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemset(gmOut, bytes, 0, bytes));

  const auto t0 = std::chrono::steady_clock::now();
  LaunchSoftmaxKernel(gmIn, gmOut, stream);
  CHECK(aclrtSynchronizeStream(stream));
  const auto t1 = std::chrono::steady_clock::now();

  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  std::printf("Kernel finished in %.3f us\n", us);

  std::vector<uint16_t> output(kTotalElements);
  CHECK(aclrtMemcpy(output.data(), bytes, gmOut, bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST));

  constexpr float kAbsTol = 5.0e-4f;
  constexpr float kRelTol = 2.0e-2f;
  constexpr double kRowSumTol = 2.0e-2;
  constexpr int kMaxPrintedErrors = 20;

  int mismatches = 0;
  int nonFiniteValues = 0;
  int rowSumFailures = 0;
  float maxAbsErr = 0.0f;
  float maxRelErr = 0.0f;

  for (int b = 0; b < kBlocks; ++b) {
    for (int group = 0; group < kSoftmaxGroupsPerBlock; ++group) {
      const int base = b * kElementsPerBlock + group * kSoftmaxLength;
      double sumGot = 0.0;
      bool groupHasNonFinite = false;

      for (int i = 0; i < kSoftmaxLength; ++i) {
        const int idx = base + i;
        const float got = HalfToFloat(output[idx]);
        const float expected = HalfToFloat(golden[idx]);

        if (!std::isfinite(got)) {
          groupHasNonFinite = true;
          ++nonFiniteValues;
          if (nonFiniteValues <= kMaxPrintedErrors) {
            std::printf("Non-finite output b=%d group=%d i=%d: got=%g\n", b,
                        group, i, got);
          }
          continue;
        }

        sumGot += got;
        const float absErr = std::fabs(got - expected);
        const float relErr = absErr / std::max(std::fabs(expected), 1.0e-8f);
        maxAbsErr = std::max(maxAbsErr, absErr);
        maxRelErr = std::max(maxRelErr, relErr);
        if (absErr > kAbsTol && relErr > kRelTol) {
          ++mismatches;
          if (mismatches <= kMaxPrintedErrors) {
            std::printf("Mismatch b=%d group=%d i=%d: got=%g expected=%g "
                        "absErr=%g relErr=%g\n",
                        b, group, i, got, expected, absErr, relErr);
          }
        }
      }

      if (!groupHasNonFinite && std::fabs(sumGot - 1.0) > kRowSumTol) {
        ++rowSumFailures;
        std::printf("Row-sum failure b=%d group=%d: got %.8f\n", b, group,
                    sumGot);
      }
    }
  }

  std::printf("Verification summary: mismatches=%d nonFinite=%d "
              "rowSumFailures=%d maxAbsErr=%g maxRelErr=%g\n",
              mismatches, nonFiniteValues, rowSumFailures, maxAbsErr,
              maxRelErr);

  const bool passed =
      mismatches == 0 && nonFiniteValues == 0 && rowSumFailures == 0;
  std::printf("Verification %s\n", passed ? "PASSED" : "FAILED");

  CHECK(aclrtFree(gmOut));
  CHECK(aclrtFree(gmIn));
  CHECK(aclrtDestroyStream(stream));
  CHECK(aclrtResetDevice(0));
  CHECK(aclFinalize());

  return passed ? 0 : 1;
}
