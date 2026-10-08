CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fPIC -pthread
UNAME_S  := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  SHLIB = libcertskip.dylib
  SHFLAGS = -dynamiclib
else
  SHLIB = libcertskip.so
  SHFLAGS = -shared
endif

all: $(SHLIB) certskip

$(SHLIB): certskip.cpp certskip.hpp rtseg.hpp
	$(CXX) $(CXXFLAGS) $(SHFLAGS) -o $@ certskip.cpp

certskip: main.cpp certskip.hpp
	$(CXX) $(CXXFLAGS) -o $@ main.cpp

clean:
	rm -f $(SHLIB) libcertskip.so libcertskip.dylib certskip

.PHONY: all clean
