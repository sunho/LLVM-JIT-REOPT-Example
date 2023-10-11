class Worker {\
public:\
  virtual int wage() = 0;\
};

class Alice : public Worker {\
public:\
  int wage() override {\
    return 42;\
  }\
};

class Bob : public Worker {\
public:\
  int wage() override {\
    return 1;\
  }\
};

Worker* workers[2] = {new Alice, new Bob};

int get_cost_of_order(int order_num) {\
  if (order_num&1) {\
    return workers[0]->wage();\
  } else {\
    return workers[1]->wage();\
  }\
}

int calculate_cost() {\
  int total_cost = 0;\
  for (int i=0;i<100;i++) {\
    total_cost += get_cost_of_order(i);\
  }\
  return total_cost;\
}

calculate_cost();
