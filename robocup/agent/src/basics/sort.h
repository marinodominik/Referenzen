#ifndef _SORT_H_
#define _SORT_H_


class Sort{
 private:
  struct SortType{
    int key;
    float value;
  };
  SortType *sortarray;
  int sortarray_size, sortarray_maxsize;
 public:
  Sort(int number);
  ~Sort() {
    delete[] sortarray;
  };
  void add(int key, float value);
  void do_sort(int type = 0); //type 0 : smallest value first
  void reset();
  int get_key(int pos);
  float get_value(int pos);
};
#endif
