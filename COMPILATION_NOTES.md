## Compilation notes ZPE5 machine

Open PowerShell VS2022

cmake -S src/main/cpp -B src/main/cpp/build -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DCMAKE_GENERATOR_TOOLSET="cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
cmake --build ./src/main/cpp/build --config Release

mvn clean package -DgenerateJniHeaders=true
