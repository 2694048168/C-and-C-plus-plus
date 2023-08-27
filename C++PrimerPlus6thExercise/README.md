### C++ Primer Plus 第6版中文版 编程练习解析

### Overview
- [Quick start](#quick-start)
- [开始学习C++ 2.7 编程练习](#27-编程练习)
- [处理数据 3.7 编程练习](#37-编程练习)
- [复合类型 4.13 编程练习](#413-编程练习)
- [循环和关系表达式 5.9 编程练习](#59-编程练习)
- [分支语句与逻辑运算符 6.11 编程练习](#611-编程练习)
- [函数-C++的编程模块 7.13 编程练习](#713-编程练习)
- [函数探幽 8.8 编程练习](#88-编程练习)
- [内存模型和名称空间 9.6 编程练习](#96-编程练习)
- [对象和类 10.10 编程练习](#1010-编程练习)
- [使用类 11.9 编程练习](#119-编程练习)
- [类和动态内存分配 12.10 编程练习](#1210-编程练习)

#### Quick start

```shell
# git clone this repo. into local path
git clone --recursive https://github.com/2694048168/C-and-C-plus-plus.git

cd C-and-C-plus-plus/C++PrimerPlus6thExercise
# install the C++ complier, such GCC, Clang, cl(MSVC) from Microsoft

# goto the folder for chapter of book and compile&run the source code
cd chapter02
clang++ 2_7_1_show_name_address.cpp -std=c++17
./a.exe
```

#### 2.7 编程练习

```shell
mkdir chapter02
cd chapter02

# 1. solution:
touch 2_7_1_show_name_address.cpp
g++ 2_7_1_show_name_address.cpp -std=c++17
clang++ 2_7_1_show_name_address.cpp -std=c++17
./a.exe

# 2. solution:
touch 2_7_2_unit_conversion.cpp
g++ 2_7_2_unit_conversion.cpp -std=c++17
clang++ 2_7_2_unit_conversion.cpp -std=c++17
./a.exe

# 3. solution:
touch 2_7_3_function.cpp
g++ 2_7_3_function.cpp -std=c++17
clang++ 2_7_3_function.cpp -std=c++17
./a.exe

# 4. solution:
touch 2_7_4_age_conversion.cpp
g++ 2_7_4_age_conversion.cpp -std=c++17
clang++ 2_7_4_age_conversion.cpp -std=c++17
./a.exe

# 5. solution:
touch 2_7_5_temperature_conversion.cpp
g++ 2_7_5_temperature_conversion.cpp -std=c++17
clang++ 2_7_5_temperature_conversion.cpp -std=c++17
./a.exe

# 6. solution:
touch 2_7_6_light_years.cpp
g++ 2_7_6_light_years.cpp -std=c++17
clang++ 2_7_6_light_years.cpp -std=c++17
./a.exe

# 7. solution:
touch 2_7_7_hour_minute.cpp
g++ 2_7_7_hour_minute.cpp -std=c++17
clang++ 2_7_7_hour_minute.cpp -std=c++17
./a.exe
```

#### 3.7 编程练习

```shell
mkdir chapter03
cd chapter03

# 1. solution:
touch 3_7_1_height_unit.cpp
g++ 3_7_1_height_unit.cpp -std=c++17
clang++ 3_7_1_height_unit.cpp -std=c++17
./a.exe

# 2. solution:
touch 3_7_2_body_mass_index.cpp
g++ 3_7_2_body_mass_index.cpp -std=c++17
clang++ 3_7_2_body_mass_index.cpp -std=c++17
./a.exe

# 3. solution:
touch 3_7_3_latitude_conversion.cpp
g++ 3_7_3_latitude_conversion.cpp -std=c++17
clang++ 3_7_3_latitude_conversion.cpp -std=c++17
./a.exe

# 4. solution:
touch 3_7_4_time_conversion.cpp
g++ 3_7_4_time_conversion.cpp -std=c++17
clang++ 3_7_4_time_conversion.cpp -std=c++17
./a.exe

# 5. solution:
touch 3_7_5_population_percentage.cpp
g++ 3_7_5_population_percentage.cpp -std=c++17
clang++ 3_7_5_population_percentages.cpp -std=c++17
./a.exe

# 6. solution:
touch 3_7_6_driving_distance.cpp
g++ 3_7_6_driving_distance.cpp -std=c++17
clang++ 3_7_6_driving_distance.cpp -std=c++17
./a.exe

# 7. solution:
touch 3_7_7_driving_conversion.cpp
g++ 3_7_7_driving_conversion.cpp -std=c++17
clang++ 3_7_7_driving_conversion.cpp -std=c++17
./a.exe

```

#### 4.13 编程练习

```shell
mkdir chapter04
cd chapter04

# 1. solution:
touch 4_13_1_show_information.cpp
g++ 4_13_1_show_information.cpp -std=c++17
clang++ 4_13_1_show_information.cpp -std=c++17
./a.exe

# 2. solution:
touch 4_13_2_string_char.cpp
g++ 4_13_2_string_char.cpp -std=c++17
clang++ 4_13_2_string_char.cpp -std=c++17
./a.exe

# 3. solution:
touch 4_13_3_person_information.cpp
g++ 4_13_3_person_information.cpp -std=c++17
clang++ 4_13_3_person_information.cpp -std=c++17
./a.exe

# 4. solution:
touch 4_13_4_person_information.cpp
g++ 4_13_4_person_information.cpp -std=c++17
clang++ 4_13_4_person_information.cpp -std=c++17
./a.exe

# 5. solution:
touch 4_13_5_structure_info.cpp
g++ 4_13_5_structure_info.cpp -std=c++17
clang++ 4_13_5_structure_info.cpp -std=c++17
./a.exe

# 6. solution:
touch 4_13_6_structure_array.cpp
g++ 4_13_6_structure_array.cpp -std=c++17
clang++ 4_13_6_structure_array.cpp -std=c++17
./a.exe

# 7. solution:
touch 4_13_7_pizza_analysis_service.cpp
g++ 4_13_7_pizza_analysis_service.cpp -std=c++17
clang++ 4_13_7_pizza_analysis_service.cpp -std=c++17
./a.exe

# 8. solution:
touch 4_13_8_pizza_analysis_service.cpp
g++ 4_13_8_pizza_analysis_service.cpp -std=c++17
clang++ 4_13_8_pizza_analysis_service.cpp -std=c++17
./a.exe

# 9. solution:
touch 4_13_9_structure_array.cpp
g++ 4_13_9_structure_array.cpp -std=c++17
clang++ 4_13_9_structure_array.cpp -std=c++17
./a.exe

# 10. solution:
touch 4_13_10_sprint_result.cpp
g++ 4_13_10_sprint_result.cpp -std=c++17
clang++ 4_13_10_sprint_result.cpp -std=c++17
./a.exe

```

#### 5.9 编程练习

```shell
mkdir chapter05
cd chapter05

# 1. solution:
touch 5_9_1_loop_sum.cpp
g++ 5_9_1_loop_sum.cpp -std=c++17
clang++ 5_9_1_loop_sum.cpp -std=c++17
./a.exe

# 2. solution:
touch 5_9_2_for_factorials.cpp
g++ 5_9_2_for_factorials.cpp -std=c++17
clang++ 5_9_2_for_factorials.cpp -std=c++17
./a.exe

# 3. solution:
touch 5_9_3_accumulative_sum.cpp
g++ 5_9_3_accumulative_sum.cpp -std=c++17
clang++ 5_9_3_accumulative_sum.cpp -std=c++17
./a.exe

# 4. solution:
touch 5_9_4_investment_value.cpp
g++ 5_9_4_investment_value.cpp -std=c++17
clang++ 5_9_4_investment_value.cpp -std=c++17
./a.exe

# 5. solution:
touch 5_9_5_total_sales.cpp
g++ 5_9_5_total_sales.cpp -std=c++17
clang++ 5_9_5_total_sales.cpp -std=c++17
./a.exe

# 6. solution:
touch 5_9_6_total_sales.cpp
g++ 5_9_6_total_sales.cpp -std=c++17
clang++ 5_9_6_total_sales.cpp -std=c++17
./a.exe

# 7. solution:
touch 5_9_7_car_info.cpp
g++ 5_9_7_car_info.cpp -std=c++17
clang++ 5_9_7_car_info.cpp -std=c++17
./a.exe

# 8. solution:
touch 5_9_8_char_processing.cpp
g++ 5_9_8_char_processing.cpp -std=c++17
clang++ 5_9_8_char_processing.cpp -std=c++17
./a.exe

# 9. solution:
touch 5_9_9_string_processing.cpp
g++ 5_9_9_string_processing.cpp -std=c++17
clang++ 5_9_9_string_processing.cpp -std=c++17
./a.exe

# 10. solution:
touch 5_9_10_nesting_loop.cpp
g++ 5_9_10_nesting_loop.cpp -std=c++17
clang++ 5_9_10_nesting_loop.cpp -std=c++17
./a.exe

```

#### 6.11 编程练习

```shell
mkdir chapter06 && cd chapter06

# 1. solution:
touch 6_11_1_uppercase_lowercase.cpp
g++ 6_11_1_uppercase_lowercase.cpp -std=c++17
clang++ 6_11_1_uppercase_lowercase.cpp -std=c++17
./a.exe

# 2. solution:
touch 6_11_2_donation_statistics.cpp
g++ 6_11_2_donation_statistics.cpp -std=c++17
clang++ 6_11_2_donation_statistics.cpp -std=c++17
./a.exe

# 3. solution:
touch 6_11_3_menu_system.cpp
g++ 6_11_3_menu_system.cpp -std=c++17
clang++ 6_11_3_menu_system.cpp -std=c++17
./a.exe

# 4. solution:
touch 6_11_4_programmers_info.cpp
g++ 6_11_4_programmers_info.cpp -std=c++17
clang++ 6_11_4_programmers_info.cpp -std=c++17
./a.exe

# 5. solution:
touch 6_11_5_tax_code.cpp
g++ 6_11_5_tax_code.cpp -std=c++17
clang++ 6_11_5_tax_code.cpp -std=c++17
./a.exe

# 6. solution:
touch 6_11_6_track_contributions.cpp
g++ 6_11_6_track_contributions.cpp -std=c++17
clang++ 6_11_6_track_contributions.cpp -std=c++17
./a.exe

# 7. solution:
touch 6_11_7_alpha_classification.cpp
g++ 6_11_7_alpha_classification.cpp -std=c++17
clang++ 6_11_7_alpha_classification.cpp -std=c++17
./a.exe

# 8. solution:
touch 6_11_8_read_file.cpp
echo "wei li c plus plus" >> test.txt
g++ 6_11_8_read_file.cpp -std=c++17
clang++ 6_11_8_read_file.cpp -std=c++17
./a.exe

# 9. solution:
touch 6_11_9_track_contributions.cpp
echo "4
Sam Stone
2000
Freida Fls
100500
Tammy Tub
5000
Rich Raptor
55000" >> Patrons.txt
g++ 6_11_9_track_contributions.cpp -std=c++17
clang++ 6_11_9_track_contributions.cpp -std=c++17
./a.exe

```

#### 7.13 编程练习

```shell
mkdir chapter07 && cd chapter07

# 1. solution:
touch 7_13_1_harmonic_mean.cpp
g++ 7_13_1_harmonic_mean.cpp -std=c++17
clang++ 7_13_1_harmonic_mean.cpp -std=c++17
./a.exe

# 2. solution:
touch 7_13_2_golf_score.cpp
g++ 7_13_2_golf_score.cpp -std=c++17
clang++ 7_13_2_golf_score.cpp -std=c++17
./a.exe

# 3. solution:
touch 7_13_3_parameter_pass.cpp
g++ 7_13_3_parameter_pass.cpp -std=c++17
clang++ 7_13_3_parameter_pass.cpp -std=c++17
./a.exe

# 4. solution:
touch 7_13_4_simple_lottery.cpp
g++ 7_13_4_simple_lottery.cpp -std=c++17
clang++ 7_13_4_simple_lottery.cpp -std=c++17
./a.exe

# 5. solution:
touch 7_13_5_recursive_function.cpp
g++ 7_13_5_recursive_function.cpp -std=c++17
clang++ 7_13_5_recursive_function.cpp -std=c++17
./a.exe

# 6. solution:
touch 7_13_6_array_operator.cpp
g++ 7_13_6_array_operator.cpp -std=c++17
clang++ 7_13_6_array_operator.cpp -std=c++17
./a.exe

# 7. solution:
touch 7_13_7_array_function.cpp
g++ 7_13_7_array_function.cpp -std=c++17
clang++ 7_13_7_array_function.cpp -std=c++17
./a.exe

# 8. solution:
touch 7_13_8_array_function.cpp
g++ 7_13_8_array_function.cpp -std=c++17
clang++ 7_13_8_array_function.cpp -std=c++17
./a.exe

# 9. solution:
touch 7_13_9_array_structure.cpp
g++ 7_13_9_array_structure.cpp -std=c++17
clang++ 7_13_9_array_structure.cpp -std=c++17
./a.exe

# 10. solution:
touch 7_13_10_callback_function.cpp
g++ 7_13_10_callback_function.cpp -std=c++17
clang++ 7_13_10_callback_function.cpp -std=c++17
./a.exe
```

#### 8.8 编程练习

```shell
mkdir chapter08 && cd chapter08

# 1. solution:
touch 8_8_1_function_call.cpp
g++ 8_8_1_function_call.cpp -std=c++17
clang++ 8_8_1_function_call.cpp -std=c++17
./a.exe

# 2. solution:
touch 8_8_2_function_pass.cpp
g++ 8_8_2_function_pass.cpp -std=c++17
clang++ 8_8_2_function_pass.cpp -std=c++17
./a.exe

# 3. solution:
touch 8_8_3_string_processing.cpp
g++ 8_8_3_string_processing.cpp -std=c++17
clang++ 8_8_3_string_processing.cpp -std=c++17
./a.exe

# 4. solution:
touch 8_8_4_string_structure.cpp
g++ 8_8_4_string_structure.cpp -std=c++17
clang++ 8_8_4_string_structure.cpp -std=c++17
./a.exe

# 5. solution:
touch 8_8_5_template_function.cpp
g++ 8_8_5_template_function.cpp -std=c++17
clang++ 8_8_5_template_function.cpp -std=c++17
./a.exe

# 6. solution:
touch 8_8_6_template_specialization.cpp
g++ 8_8_6_template_specialization.cpp -std=c++17
clang++ 8_8_6_template_specialization.cpp -std=c++17
./a.exe

# 7. solution:
touch 8_8_7_template.cpp
g++ 8_8_7_template.cpp -std=c++17
clang++ 8_8_7_template.cpp -std=c++17
./a.exe

```

#### 9.6 编程练习

```shell
mkdir chapter09 && cd chapter09

# 1. solution:
touch 9_6_1_main.cpp
touch 9_6_1_golf.hpp
touch 9_6_1_golf.cpp
g++ 9_6_1_main.cpp 9_6_1_golf.cpp -std=c++17
clang++ 9_6_1_main.cpp 9_6_1_golf.cpp -std=c++17
./a.exe

# 2. solution:
touch 9_6_2_character_count.cpp
g++ 9_6_2_character_count.cpp -std=c++17
clang++ 9_6_2_character_count.cpp -std=c++17
./a.exe

# 3. solution:
touch 9_6_3_memory_model.cpp
g++ 9_6_3_memory_model.cpp -std=c++17
clang++ 9_6_3_memory_model.cpp -std=c++17
./a.exe

# 4. solution:
touch 9_6_4_main.cpp
touch 9_6_4_namespace.cpp
touch 9_6_4_namespace.hpp
g++ 9_6_4_main.cpp 9_6_4_namespace.cpp -std=c++17
clang++ 9_6_4_main.cpp 9_6_4_namespace.cpp -std=c++17
./a.exe

# CMake 管理工程文件和C++编译和链接
touch CMakeLists.txt
cmake -S . -B build -G Ninja
cmake --build build
./build/solution1.exe
./build/solution4.exe

```

#### 10.10 编程练习

```shell
mkdir chapter10 && cd chapter10

# 1. solution:
touch 10_10_1_class_object.cpp
g++ 10_10_1_class_object.cpp -std=c++17
clang++ 10_10_1_class_object.cpp -std=c++17
./a.exe

# 2. solution:
touch 10_10_2_constructor.cpp
touch 10_10_2_person.hpp
touch 10_10_2_person.cpp
g++ 10_10_2_constructor.cpp 10_10_2_person.cpp -std=c++17
clang++ 10_10_2_constructor.cpp 10_10_2_person.cpp -std=c++17
./a.exe

# 3. solution:
touch 10_10_3_golf_class.cpp
g++ 10_10_3_golf_class.cpp -std=c++17
clang++ 10_10_3_golf_class.cpp -std=c++17
./a.exe

# 4. solution:
touch 10_10_4_main.cpp
touch 10_10_4_sale_class.hpp
touch 10_10_4_sale_class.cpp
g++ 10_10_4_main.cpp 10_10_4_sale_class.cpp -std=c++17
clang++ 10_10_4_main.cpp 10_10_4_sale_class.cpp -std=c++17
./a.exe

# 5. solution:
touch 10_10_5_stack.cpp
g++ 10_10_5_stack.cpp -std=c++17
clang++ 10_10_5_stack.cpp -std=c++17
./a.exe

# 6. solution:
touch 10_10_6_2d_move.cpp
g++ 10_10_6_2d_move.cpp -std=c++17
clang++ 10_10_6_2d_move.cpp -std=c++17
./a.exe

# 7. solution:
touch 10_10_7_class.cpp
g++ 10_10_7_class.cpp -std=c++17
clang++ 10_10_7_class.cpp -std=c++17
./a.exe

# 8. solution:
touch 10_10_8_main_adt.cpp
touch 10_10_8_list_adt.hpp
touch 10_10_8_list_adt.cpp
g++ 10_10_8_main_adt.cpp 10_10_8_list_adt.cpp -std=c++17
clang++ 10_10_8_main_adt.cpp 10_10_8_list_adt.cpp -std=c++17
./a.exe

```

#### 11.9 编程练习

```shell
mkdir chapter11 && cd chapter11

touch read_write_file.cpp

# 1. solution:
touch 11_9_1_random_walk.cpp
touch 11_9_1_my_vector.hpp
touch 11_9_1_my_vector.cpp
g++ 11_9_1_random_walk.cpp 11_9_1_my_vector.cpp -std=c++17
clang++ 11_9_1_random_walk.cpp 11_9_1_my_vector.cpp -std=c++17
./a.exe

# 2. solution:
touch 11_9_2_random_walk.cpp
touch 11_9_2_my_vector_modify.hpp
touch 11_9_2_my_vector_modify.cpp
g++ 11_9_2_random_walk.cpp 11_9_2_my_vector_modify.cpp -std=c++17
clang++ 11_9_2_random_walk.cpp 11_9_2_my_vector_modify.cpp -std=c++17
./a.exe

# 3. solution:
touch 11_9_3_random_walk_report.cpp
g++ 11_9_3_random_walk_report.cpp 11_9_1_my_vector.cpp -std=c++17
clang++ 11_9_3_random_walk_report.cpp 11_9_1_my_vector.cpp -std=c++17
./a.exe

# 4. solution:
touch 11_9_4_main_time.cpp
touch 11_9_4_time_class.hpp
touch 11_9_4_time_class.cpp
g++ 11_9_4_main_time.cpp 11_9_4_time_class.cpp -std=c++17
clang++ 11_9_4_main_time.cpp 11_9_4_time_class.cpp -std=c++17
./a.exe

# 5. solution:
touch 11_9_5_main_stone.cpp
touch 11_9_5_stone_weight.hpp
touch 11_9_5_stone_weight.cpp
g++ 11_9_5_main_stone.cpp 11_9_5_stone_weight.cpp -std=c++17
clang++ 11_9_5_main_stone.cpp 11_9_5_stone_weight.cpp -std=c++17
./a.exe

# 6. solution:
touch 11_9_6_main.cpp
touch 11_9_6_stone_weight_overload.hpp
touch 11_9_6_stone_weight_overload.cpp
g++ 11_9_6_main.cpp 11_9_6_stone_weight_overload.cpp -std=c++17
clang++ 11_9_6_main.cpp 11_9_6_stone_weight_overload.cpp -std=c++17
./a.exe

# 7. solution:
touch 11_9_7_main_complex.cpp
touch 11_9_7_my_complex.hpp
touch 11_9_7_my_complex.cpp
g++ 11_9_7_main_complex.cpp 11_9_7_my_complex.cpp -std=c++17
clang++ 11_9_7_main_complex.cpp 11_9_7_my_complex.cpp -std=c++17
./a.exe

```

#### 12.10 编程练习

```shell
mkdir chapter12 && cd chapter12

# 1. solution:
touch 12_10_1_cow_class.cpp
g++ 12_10_1_cow_class.cpp -std=c++17
clang++ 12_10_1_cow_class.cpp -std=c++17
./a.exe

# 2. solution:
touch 12_10_2_test_string.cpp
touch 12_10_2_my_string.hpp
touch 12_10_2_my_string.cpp
g++ 12_10_2_test_string.cpp 12_10_2_my_string.cpp -std=c++17
clang++ 12_10_2_test_string.cpp 12_10_2_my_string.cpp -std=c++17
./a.exe

# 3. solution:
touch 12_10_3_main.cpp
touch 12_10_3_stock.hpp
touch 12_10_3_stock.cpp
g++ 12_10_3_main.cpp 12_10_3_stock.cpp -std=c++17
clang++ 12_10_3_main.cpp 12_10_3_stock.cpp -std=c++17
./a.exe

# 4. solution:
touch 12_10_4_main.cpp
touch 12_10_4_stack.hpp
touch 12_10_4_stack.cpp
g++ 12_10_4_main.cpp 12_10_4_stack.cpp -std=c++17
clang++ 12_10_4_main.cpp 12_10_4_stack.cpp -std=c++17
./a.exe

# 5. solution:TODO
# 6. solution:TODO

```

#### 13.11 编程练习

```shell
mkdir chapter13 && cd chapter13

# 1. solution:
# std::strcpy ---> strcpy_s
touch 13_11_1_main.cpp
touch 13_11_1_base.hpp
touch 13_11_1_classic.cpp
g++ 13_11_1_main.cpp 13_11_1_classic.cpp -std=c++17
clang++ 13_11_1_main.cpp 13_11_1_classic.cpp -std=c++17
./a.exe


```
