// Test runner. An optional argument filters tests by substring.

#include <chrono>
#include <cstring>
#include <exception>

#include "test_util.hpp"

int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  int passed = 0, failed = 0;
  for (const auto& t : test_registry()) {
    if (filter && !std::strstr(t.name, filter)) continue;
    auto t0 = std::chrono::steady_clock::now();
    try {
      t.fn();
      double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::cout << "[ PASS ] " << t.name << " (" << static_cast<long>(ms) << " ms)" << std::endl;
      ++passed;
    } catch (const TestFailure& f) {
      std::cout << "[ FAIL ] " << t.name << "\n         " << f.msg << std::endl;
      ++failed;
    } catch (const std::exception& e) {
      std::cout << "[ FAIL ] " << t.name << "\n         unexpected exception: " << e.what() << std::endl;
      ++failed;
    }
  }
  std::cout << "\n" << passed << " passed, " << failed << " failed" << std::endl;
  return failed == 0 ? 0 : 1;
}
