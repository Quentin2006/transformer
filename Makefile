CXX ?= g++
CPPFLAGS += -Iinclude -Ithird_party/eigen
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS ?=

TARGET := transformer
SRCS := main.cpp src/nn.cpp
OBJDIR := objects
DEPDIR := dependencies
OBJS := $(addprefix $(OBJDIR)/,$(SRCS:.cpp=.o))
DEPS := $(addprefix $(DEPDIR)/,$(SRCS:.cpp=.d))

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
	mkdir -p $(dir $@) $(dir $(DEPDIR)/$*.d)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF $(DEPDIR)/$*.d -c $< -o $@

-include $(DEPS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJDIR) $(DEPDIR) $(TARGET)
