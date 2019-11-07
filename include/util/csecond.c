#include <sys/types.h>
#include <sys/time.h>

double csecond()
{
        struct timeval  tv;
        gettimeofday(&tv, 0);
        return tv.tv_sec + tv.tv_usec * 1.0e-6;
}
