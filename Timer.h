#include <chrono>

class Timer
{
  public:
    Timer()
        : now_(std::chrono::steady_clock::now())
    {}

    float elapsed_ms() const { return elapsed_us() / 1e3; }

    float elapsed_us() const
    {
        auto end = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - now_).count();
    }

    float elapsed_sec() const { return elapsed_us() / 1e6; }

  private:
    std::chrono::steady_clock::time_point now_;
};
