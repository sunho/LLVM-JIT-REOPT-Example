#include <stdlib.h>

void this_is_actually_just_no_op() {\
  int unused_var = 0;\
  for (int i=0;i<1000;i++){\
    unused_var += i;\
  }\
}

void run_it_many_times() {\
  for (int i=0;i<1e8;i++) {\
    this_is_actually_just_no_op();\
  }\
}
run_it_many_times();
exit(0);
