//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <time.h>
#include <fstream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/platform/enforce.h"

/*
 * Manually set call_stack_level before use C++ code base
 */
DEFINE_int32(call_stack_level, 2, "print stacktrace");
using namespace fLI;

namespace paddle {
namespace train {

// also defined in paddle::inference
void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);

  PADDLE_ENFORCE_EQ(
      fin.is_open(), true,
      platform::errors::Unavailable("Failed to open file %s.", filename));

  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}


// also see fluid/inference/io.cc
std::unique_ptr<paddle::framework::ProgramDesc> Load(
    paddle::framework::Executor* executor, const std::string& model_filename) {
  VLOG(3) << "loading model from " << model_filename;
  std::string program_desc_str;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<paddle::framework::ProgramDesc> program(
      new paddle::framework::ProgramDesc(program_desc_str));
  return program;
}

}  // namespace train
}  // namespace paddle

#define EPOCH 4
#define BATCH_SIZE 2
#define FEATURE_DIM 13
#define OUTPUT_DIM 1

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  paddle::framework::InitDevices();

  const auto cpu_place = paddle::platform::CPUPlace();

  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;

  // init program
  auto startup_program = paddle::train::Load(&executor, "startup_program");
  auto train_program = paddle::train::Load(&executor, "main_program");

  std::string loss_name = "";
  for (auto op_desc : train_program->Block(0).AllOps()) {
    std::cout << "Op_desc : " << op_desc->Type() << " ";
    if (op_desc->Type() == "mean") {
      loss_name = op_desc->Output("Out")[0];
      break;
    }
  }
  std::cout << std::endl;

  std::cout << "loss_name: " << loss_name << std::endl;

  PADDLE_ENFORCE_NE(loss_name, "",
                    paddle::platform::errors::NotFound("Loss name is not found."));

  // init all parameters
  executor.Run(*startup_program, &scope, 0);

  // get feed target names and fetch target names


  // prepare data
  float train_X[][FEATURE_DIM] = {
      {-0.01475919, -0.11363636,  0.30950225, -0.06916996,  0.10350811,
          0.00658471,  0.28347167, -0.15413823, -0.19780031, -0.00999457,
          -0.39952485, -0.15024467, -0.04285495},
      {-0.03904606, -0.11363636, -0.0944567 , -0.06916996, -0.07138901,
          -0.02253964,  0.22064983, -0.12494818, -0.19780031, -0.04625411,
          0.26004962,  0.0908164 , -0.00891455},
      {-0.03991703,  0.28636364, -0.36241857, -0.06916996, -0.25863181,
          0.03934961, -0.24896912,  0.454406  , -0.37171335, -0.13976556,
          0.13239004,  0.10143217, -0.1841353},
      {-0.04049195,  0.43636364, -0.32576168, -0.06916996, -0.34093634,
          0.03226013, -0.37770238,  0.31938613, -0.37171335, -0.20655945,
          -0.33569506,  0.09593517, -0.12204921},
      // 4
      {-0.03924343, -0.11363636, -0.30230127, -0.06916996, -0.22571   ,
          0.06521663, -0.11096706, -0.02726611, -0.32823509, -0.25236098,
          -0.04846102,  0.00329308, -0.16564744},
      {-0.02379686, -0.11363636,  0.30950225, -0.06916996,  0.65083321,
          -0.16873623,  0.32363645, -0.20036944, -0.19780031, -0.00999457,
          -0.39952485, -0.03801007,  0.01729958},
      {-0.03646541, -0.11363636, -0.04533646, -0.06916996, -0.02200629,
          0.05410339,  0.19284345, -0.017527  , -0.24127857, -0.19892587,
          -0.00590783,  0.09838108, -0.0930757},
      {-0.03192691, -0.11363636, -0.10985259, -0.06916996, -0.03435197,
          0.04030765,  0.26596394,  0.05998575, -0.24127857, -0.19320068,
          0.27068792,  0.07883899,  0.00405455}

  };

  float train_y[][OUTPUT_DIM] = {
      {23.8},
      {20.1},
      {22.9},
      {22.},
      // 4
      {28.4},
      {19.6},
      {23.8},
      {18.4}

  };

  auto x_var = scope.Var("x");
  auto x_tensor = x_var->GetMutable<paddle::framework::LoDTensor>();
  x_tensor->Resize({BATCH_SIZE, FEATURE_DIM});

  auto x_data = x_tensor->mutable_data<float>(cpu_place);
  for (int i = 0; i < BATCH_SIZE * FEATURE_DIM; ++i) {
    float* train_data = &train_X[0][0];
    x_data[i] = static_cast<float>( train_data[i] );
  }

  auto y_var = scope.Var("y");
  auto y_tensor = y_var->GetMutable<paddle::framework::LoDTensor>();
  y_tensor->Resize({BATCH_SIZE, OUTPUT_DIM});
  auto y_data = y_tensor->mutable_data<float>(cpu_place);
  for (int i = 0; i < BATCH_SIZE * OUTPUT_DIM; ++i) {
    float* train_logit = &train_y[0][0];
    y_data[i] = static_cast<float>( train_logit[i] );
  }

  auto loss_var = scope.Var(loss_name);
  auto loss_tensor = loss_var->GetMutable<paddle::framework::LoDTensor>();
  loss_tensor->Resize({OUTPUT_DIM, OUTPUT_DIM});
  auto loss_data = loss_tensor->mutable_data<float>(cpu_place);

  paddle::platform::ProfilerState pf_state;
  pf_state = paddle::platform::ProfilerState::kCPU;
  paddle::platform::EnableProfiler(pf_state);
  clock_t t1 = clock();

  for (int i = 0; i < 10; ++i) {
    /*
     * see reference source code in
     * /home/yiakwy/anaconda3/envs/py36/lib/python3.6/site-packages/paddle/fluid/executor.py:1327
     * When to use program_cache by setting `create_local_scope` and `create_vars` values
     */
    auto loss_tensor = loss_var->GetMutable<paddle::framework::LoDTensor>();
    loss_tensor->Resize({1, 1});
    auto loss_data = loss_tensor->mutable_data<float>(cpu_place);
    executor.Run(*train_program, &scope, 0, false, true, std::vector<std::basic_string<char>>(), true);

    // fetch output tensor value
    std::cout << "step: " << i << " loss: "
              << loss_var->Get<paddle::framework::LoDTensor>().data<float>()[0]
              // << loss_data[0]
              << std::endl;
  }

  clock_t t2 = clock();
  paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                    "run_paddle_op_profiler");
  std::cout << "run_time = " << t2 - t1 << std::endl;
  return 0;
}
