TORCHDIR = /Users/blakeduschatko/CodeLibraries/libtorch/

FC = gfortran
CXX = g++

UNAME := $(shell uname -s)

FCFLAGS = -Wall -Wextra
CCFLAGS = -Wall -Wextra -std=c++11 -isystem $(TORCHDIR)include -isystem $(TORCHDIR)include/torch/csrc/api/include
ifeq ($(UNAME_S),Darwin)
	LDFLAGS = -lstdc++ -Wl,-rpath,$(TORCHDIR)lib $(TORCHDIR)lib/libtorch.dylib $(TORCHDIR)lib/libcaffe2.dylib $(TORCHDIR)lib/libc10.dylib
else
	LDFLAGS = -lc++ -Wl,-rpath,$(TORCHDIR)lib $(TORCHDIR)lib/libtorch.dylib $(TORCHDIR)lib/libcaffe2.dylib $(TORCHDIR)lib/libc10.dylib
endif

all: main.x

%.x : main.o square.o
	${FC} $^ -o $@ ${LDFLAGS}

%.o : %.f90
	${FC} -c $< -o $@

%.o : %.cpp
	${CXX} ${CCFLAGS} -c $^ -o $@

.PHONY : clean

clean :
	${RM} -rf *.o *.mod main.x
