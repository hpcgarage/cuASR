#!/usr/bin/sh
find ./src     -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -style=file -i {} \;
find ./include -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -style=file -i {} \;
find ./bench   -maxdepth 1 -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -style=file -i {} \;
find ./test    -maxdepth 1 -regex '.*\.\(cpp\|hpp\|cu\|cuh\)' -exec clang-format -style=file -i {} \;
