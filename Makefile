CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I./include

SRCDIR = src
INCDIR = include
OBJDIR = obj

SOURCES = $(SRCDIR)/main.cpp \
          $(SRCDIR)/sparse_tensor.cpp \
          $(SRCDIR)/dense_tensor.cpp \
          $(SRCDIR)/inference.cpp \
          $(SRCDIR)/parser.cpp \
          $(SRCDIR)/repl.cpp

OBJECTS = $(SOURCES:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

TARGET = tl
EXAMPLES = tl-examples

.PHONY: all clean test examples

all: $(TARGET) $(EXAMPLES)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(EXAMPLES): $(SRCDIR)/examples.cpp $(OBJDIR)/sparse_tensor.o $(OBJDIR)/dense_tensor.o $(OBJDIR)/inference.o $(OBJDIR)/parser.o
	$(CXX) $(CXXFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR):
	mkdir -p $(OBJDIR)

clean:
	rm -rf $(OBJDIR) $(TARGET) $(EXAMPLES)

test: $(TARGET)
	./$(TARGET) --test

examples: $(EXAMPLES)
	./$(EXAMPLES)

example-%: $(EXAMPLES)
	./$(EXAMPLES) $*

run: $(TARGET)
	./$(TARGET)

file: $(TARGET)
	./$(TARGET) examples/ancestor.tl
