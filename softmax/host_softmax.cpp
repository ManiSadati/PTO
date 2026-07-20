#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <runtime/rt.h>
#include <vector>

#define KERNEL_NAME "softmax__kernel0"
#define BLOCK_DIM 8

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

static_assert(kBlocks == BLOCK_DIM, "BLOCK_DIM must match the data shape.");
static_assert(kElementsPerBlock == 2048,
              "The device kernel expects 2048 FP16 elements per block.");

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
  if (fileSize < 0) {
    std::perror("ftell");
    std::fclose(f);
    return false;
  }

  if (static_cast<size_t>(fileSize) != expectedBytes) {
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

bool ReadWholeFile(const char *path, std::vector<char> *out) {
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
  if (fileSize <= 0) {
    std::printf("Invalid file size for %s: %ld\n", path, fileSize);
    std::fclose(f);
    return false;
  }

  std::rewind(f);
  out->resize(static_cast<size_t>(fileSize));
  const size_t nread = std::fread(out->data(), 1, out->size(), f);
  std::fclose(f);

  if (nread != out->size()) {
    std::printf("Short read for %s: expected %zu bytes, got %zu bytes\n", path,
                out->size(), nread);
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
      const uint32_t exp32 = static_cast<uint32_t>(e + 127);
      out = (sign << 31) | (exp32 << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | 0x7f800000u | (mant << 13);
  } else {
    const uint32_t exp32 = exp - 15 + 127;
    out = (sign << 31) | (exp32 << 23) | (mant << 13);
  }

  union {
    uint32_t u;
    float f;
  } v{out};
  return v.f;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::printf(
        "Usage: %s <softmax.elf> <input_fp16.bin> <golden_fp16.bin>\n",
        argv[0]);
    return 1;
  }

  std::vector<char> bin;
  if (!ReadWholeFile(argv[1], &bin)) {
    return 1;
  }

  std::vector<uint16_t> input(kTotalElements);
  std::vector<uint16_t> golden(kTotalElements);
  const size_t bytes = input.size() * sizeof(uint16_t);

  if (!ReadBinaryFile(argv[2], input.data(), bytes) ||
      !ReadBinaryFile(argv[3], golden.data(), bytes)) {
    return 1;
  }

  CHECK(aclInit(nullptr));
  CHECK(rtSetDevice(0));

  void *handle = nullptr;
  rtDevBinary_t binary{};
  binary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  binary.data = bin.data();
  binary.length = bin.size();
  CHECK(rtDevBinaryRegister(&binary, &handle));

  uint32_t stub = 0;
  CHECK(rtFunctionRegister(handle, &stub, KERNEL_NAME, KERNEL_NAME, 0));

  rtStream_t stream = nullptr;
  CHECK(rtStreamCreate(&stream, 0));

  void *gmIn = nullptr;
  void *gmOut = nullptr;
  CHECK(aclrtMalloc(&gmIn, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(aclrtMalloc(&gmOut, bytes, ACL_MEM_MALLOC_HUGE_FIRST));

  CHECK(aclrtMemcpy(gmIn, bytes, input.data(), bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE));
  CHECK(aclrtMemset(gmOut, bytes, 0, bytes));

  // rtKernelLaunch expects a packed argument buffer containing the two device
  // pointers in the same order as the kernel signature.
  struct KernelArgs {
    void *data;
    void *cast_1;
  } args{gmIn, gmOut};

  const auto t0 = std::chrono::steady_clock::now();
  CHECK(rtKernelLaunch(&stub, BLOCK_DIM, &args, sizeof(args), nullptr, stream));
  CHECK(rtStreamSynchronize(stream));
  const auto t1 = std::chrono::steady_clock::now();

  const double us =
      std::chrono::duration<double, std::micro>(t1 - t0).count();
  std::printf("Kernel finished in %.3f us\n", us);

  std::vector<uint16_t> output(kTotalElements);
  CHECK(aclrtMemcpy(output.data(), bytes, gmOut, bytes,
                    ACL_MEMCPY_DEVICE_TO_HOST));

  // The output is FP16 and the kernel uses vector exp/div plus an FP16
  // subtraction stage. These tolerances are strict enough to catch layout,
  // broadcast, and normalization failures without requiring bit equality.
  constexpr float kAbsTol = 5.0e-4f;
  constexpr float kRelTol = 2.0e-2f;
  constexpr double kRowSumTol = 2.0e-2;
  constexpr int kMaxPrintedErrors = 20;

  int mismatches = 0;
  int nonFiniteValues = 0;
  int rowSumFailures = 0;
  float maxAbsErr = 0.0f;
  float maxRelErr = 0.0f;

  int worstAbsBlock = -1;
  int worstAbsGroup = -1;
  int worstAbsIndex = -1;
  float worstAbsGot = 0.0f;
  float worstAbsExpected = 0.0f;

  for (int b = 0; b < kBlocks; ++b) {
    for (int group = 0; group < kSoftmaxGroupsPerBlock; ++group) {
      const int base = b * kElementsPerBlock + group * kSoftmaxLength;
      double sumGot = 0.0;
      double sumExpected = 0.0;
      bool groupHasNonFinite = false;

      for (int i = 0; i < kSoftmaxLength; ++i) {
        const int idx = base + i;
        const float got = HalfToFloat(output[idx]);
        const float expected = HalfToFloat(golden[idx]);
        sumExpected += expected;

        if (!std::isfinite(got)) {
          groupHasNonFinite = true;
          ++nonFiniteValues;
          if (nonFiniteValues <= kMaxPrintedErrors) {
            std::printf(
                "Non-finite output b=%d group=%d i=%d: got=%g "
                "got_hex=0x%04x\n",
                b, group, i, got, output[idx]);
          }
          continue;
        }

        sumGot += got;
        const float absErr = std::fabs(got - expected);
        const float relErr =
            absErr / std::max(std::fabs(expected), 1.0e-8f);

        if (absErr > maxAbsErr) {
          maxAbsErr = absErr;
          worstAbsBlock = b;
          worstAbsGroup = group;
          worstAbsIndex = i;
          worstAbsGot = got;
          worstAbsExpected = expected;
        }
        maxRelErr = std::max(maxRelErr, relErr);

        if (absErr > kAbsTol && relErr > kRelTol) {
          ++mismatches;
          if (mismatches <= kMaxPrintedErrors) {
            std::printf(
                "Mismatch b=%d group=%d i=%d: got=%g expected=%g "
                "absErr=%g relErr=%g got_hex=0x%04x expected_hex=0x%04x\n",
                b, group, i, got, expected, absErr, relErr, output[idx],
                golden[idx]);
          }
        }
      }

      if (!groupHasNonFinite && std::fabs(sumGot - 1.0) > kRowSumTol) {
        ++rowSumFailures;
        std::printf(
            "Row-sum failure b=%d group=%d: got %.8f, expected about 1\n",
            b, group, sumGot);
      }

      if (!std::isfinite(sumExpected) ||
          std::fabs(sumExpected - 1.0) > kRowSumTol) {
        std::printf("Golden row-sum warning b=%d group=%d: %.8f\n", b,
                    group, sumExpected);
      }
    }
  }

  std::printf(
      "Verification summary: mismatches=%d nonFinite=%d rowSumFailures=%d "
      "maxAbsErr=%g maxRelErr=%g\n",
      mismatches, nonFiniteValues, rowSumFailures, maxAbsErr, maxRelErr);

  if (worstAbsBlock >= 0) {
    std::printf(
        "Worst absolute error: b=%d group=%d i=%d got=%g expected=%g\n",
        worstAbsBlock, worstAbsGroup, worstAbsIndex, worstAbsGot,
        worstAbsExpected);
  }

  const bool passed =
      mismatches == 0 && nonFiniteValues == 0 && rowSumFailures == 0;
  std::printf("Verification %s\n", passed ? "PASSED" : "FAILED");

  CHECK(aclrtFree(gmOut));
  CHECK(aclrtFree(gmIn));
  CHECK(rtStreamDestroy(stream));
  CHECK(rtDeviceReset(0));
  CHECK(aclFinalize());

  return passed ? 0 : 1;
}
