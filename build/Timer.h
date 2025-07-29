#pragma once

#pragma once

#include <chrono>
#include <iostream>

namespace Helper {
    class Timer
    {
    public:
        Timer() : outputAtExit(true)
        {
            start = std::chrono::steady_clock::now();
        }

        void setOutputAtExit(bool _outputAtExit) {
            outputAtExit = _outputAtExit;
        }

        std::chrono::steady_clock::time_point getStart() const {
            return start;
        }

        std::chrono::milliseconds getDuration() const {
            return std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start);
        }

        ~Timer()
        {
            if (outputAtExit)
            {
                std::cout << "Destructor called: Time difference needed for program execution: " << getDuration().count() << " Milliseconds.\n";
            }
        }

    private:
        bool outputAtExit{};
        std::chrono::steady_clock::time_point start{};
    };
}
