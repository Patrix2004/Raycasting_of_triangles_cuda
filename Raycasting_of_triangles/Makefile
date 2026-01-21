APP := Raycasting_of_triangles
SRC := kernel.cu

# CUDA nvcc path (MSYS2 style)
NVCC := "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/bin/nvcc.exe"



NVCCFLAGS := -O3 -std=c++17

UNAME_S := $(shell uname -s)


ifeq ($(UNAME_S),Linux)

CXXFLAGS := $(shell pkg-config --cflags glfw3 glew)
LDFLAGS  := $(shell pkg-config --libs glfw3 glew) -lGL -ldl -lpthread

$(APP): $(SRC)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(APP)

.PHONY: clean

else


GLFW_DIR ?= C:/libs/glfw
GLEW_DIR ?= C:/libs/glew

INCLUDES := -I"$(GLFW_DIR)/include" -I"$(GLEW_DIR)/include"
LIBS     := -L"$(GLFW_DIR)/lib-mingw-w64" -L"$(GLEW_DIR)/lib/Release/x64"

LDFLAGS  := $(LIBS) -lglfw3 -lglew32 -lopengl32 -lgdi32 -luser32 -lkernel32

$(APP).exe: $(SRC)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $< $(LDFLAGS)

clean:
	del /Q $(APP).exe 2>NUL || exit 0

.PHONY: clean
endif
