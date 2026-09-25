// Offline numerical oracle only: no Robot object, network or controller activation.
#include <franka/joint_velocity_limits.h>
#include <franka/rate_limiting.h>
#include <franka/lowpass_filter.h>
#include <nlohmann/json.hpp>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
using json = nlohmann::json;
using J7 = std::array<double,7>;
int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::ifstream urdf(argv[1]), input(argv[2]);
  if (!urdf || !input) return 2;
  std::string xml((std::istreambuf_iterator<char>(urdf)), {});
  franka::JointVelocityLimitsConfig limits(xml);
  json cases; input >> cases;
  json result;
  for (const auto& c: cases.at("limiter_cases")) {
    J7 q=c.at("q_d"), v=c.at("dq_d"), a=c.at("ddq_d"), target=c.at("target");
    auto upper=limits.getUpperJointVelocityLimits(q);
    auto lower=limits.getLowerJointVelocityLimits(q);
    auto legacy_upper=franka::computeUpperLimitsJointVelocity(q);
    auto legacy_lower=franka::computeLowerLimitsJointVelocity(q);
    auto limited=franka::limitRate(upper,lower,franka::kMaxJointAcceleration,
                                 franka::kMaxJointJerk,target,q,v,a);
    J7 filtered{};
    for (size_t i=0;i<7;++i) filtered[i]=franka::lowpassFilter(.001,target[i],q[i],100.0);
    auto filter_then_limit=franka::limitRate(legacy_upper,legacy_lower,
        franka::kMaxJointAcceleration,franka::kMaxJointJerk,filtered,q,v,a);
    result["limiter_cases"].push_back({{"id",c.at("id")},{"description_upper",upper},
      {"description_lower",lower},{"legacy_upper",legacy_upper},{"legacy_lower",legacy_lower},
      {"stock_default_output",target},{"optional_description_limited",limited},
      {"optional_filter_100Hz",filtered},{"optional_stock_filter_then_legacy_limit",filter_then_limit}});
  }
  for(const auto& c: cases.at("observation_cases")) {
    std::array<float,24> obs{}; std::array<float,7> target{};
    for(size_t i=0;i<7;++i) {
      obs[i]=c.at("q")[i].get<float>()-c.at("default")[i].get<float>();
      obs[7+i]=c.at("dq")[i].get<float>();
      obs[17+i]=c.at("previous_raw")[i].get<float>();
      target[i]=c.at("default")[i].get<float>()+0.5f*c.at("action")[i].get<float>();
    }
    for(size_t i=0;i<3;++i) obs[14+i]=c.at("target_xyz")[i].get<float>()-c.at("flange_xyz")[i].get<float>();
    result["observation_cases"].push_back({{"id",c.at("id")},{"observation",obs},{"mapped_target",target}});
  }
  result["hardware_accessed"]=false;
  result["scope"]="offline helper and observation math; no deployed governor or policy inference";
  std::cout << result.dump(2) << '\n';
}
