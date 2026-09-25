#include <franka_governor/governor.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
using namespace franka_governor;
// Python binding is non-RT. Only the C++ Governor is allocation-free.
class Batch {
 public:
  std::vector<Governor> governors;
  Batch(const Config& c,size_t n) { if(n==0)throw std::invalid_argument("empty batch");governors.reserve(n);for(size_t i=0;i<n;++i)governors.emplace_back(c); }
  std::vector<Output> step(std::int64_t now,const std::vector<Feedback>& f) {
    if(f.size()!=governors.size())throw std::invalid_argument("feedback batch size mismatch");
    std::vector<Output> result;result.reserve(governors.size());
    for(size_t i=0;i<governors.size();++i)result.push_back(governors[i].step(now,f[i]));
    return result;
  }
};
PYBIND11_MODULE(_core,m) {
  m.attr("__version__")="0.1.0";m.attr("TICK_NS")=tick_ns;
  py::enum_<Status>(m,"Status").value("DISARMED",Status::Disarmed).value("RUNNING",Status::Running)
    .value("STOPPING",Status::Stopping).value("TERMINAL",Status::Terminal).value("FAULT",Status::Fault);
  py::enum_<Reason>(m,"Reason").value("NONE",Reason::None).value("REQUESTED_STOP",Reason::RequestedStop)
    .value("INVALID_ACTION",Reason::InvalidAction).value("SEQUENCE",Reason::Sequence).value("SESSION",Reason::Session)
    .value("FUTURE_TIME",Reason::FutureTime).value("STALE_ACTION",Reason::StaleAction)
    .value("STALE_OBSERVATION",Reason::StaleObservation).value("STATE_INVALID",Reason::StateInvalid)
    .value("TIMING",Reason::Timing).value("TRACKING",Reason::Tracking).value("INFEASIBLE",Reason::Infeasible)
    .value("INTERVENTION",Reason::Intervention);
#define FIELD(T,N) .def_readwrite(#N,&T::N)
  py::class_<Config>(m,"Config").def(py::init<>())
    FIELD(Config,lower) FIELD(Config,upper) FIELD(Config,margin) FIELD(Config,max_velocity)
    FIELD(Config,max_acceleration) FIELD(Config,max_jerk) FIELD(Config,envelope_velocity)
    FIELD(Config,envelope_offset) FIELD(Config,envelope_deceleration) FIELD(Config,default_position)
    FIELD(Config,tracking_error) FIELD(Config,desired_error) FIELD(Config,desired_velocity_error) FIELD(Config,desired_acceleration_error) FIELD(Config,max_segment_distance)
    FIELD(Config,max_projection) FIELD(Config,horizon_ticks) FIELD(Config,max_blocked_ticks)
    FIELD(Config,max_projected_ticks) FIELD(Config,action_timeout_ns) FIELD(Config,observation_timeout_ns)
    FIELD(Config,tick_tolerance_ns).def("validate",&Config::validate);
  py::class_<Feedback>(m,"Feedback").def(py::init<>()) FIELD(Feedback,q) FIELD(Feedback,dq)
    FIELD(Feedback,desired_q) FIELD(Feedback,desired_dq) FIELD(Feedback,desired_ddq)
    FIELD(Feedback,observed_ns) FIELD(Feedback,session) FIELD(Feedback,healthy);
  py::class_<Message>(m,"Message").def(py::init<>()) FIELD(Message,action) FIELD(Message,sequence)
    FIELD(Message,session) FIELD(Message,observation_ns) FIELD(Message,completed_ns);
#undef FIELD
#define OUT(N) .def_readonly(#N,&Output::N)
  py::class_<Output>(m,"Output") OUT(q) OUT(dq) OUT(ddq) OUT(mapped_target) OUT(projected_target)
    OUT(previous_raw) OUT(status) OUT(reason) OUT(accepted_sequence) OUT(command_valid)
    OUT(replanning_blocked) OUT(projection);
#undef OUT
  py::class_<Governor>(m,"Governor").def(py::init<const Config&>()).def("reset",&Governor::reset)
    .def("submit",&Governor::submit).def("stop",&Governor::stop,py::arg("reason")=Reason::RequestedStop)
    .def("step",[](Governor& g,std::int64_t t,const Feedback& f){return Output(g.step(t,f));})
    .def_property_readonly("output",[](const Governor& g){return Output(g.output());});
  py::class_<Batch>(m,"Batch").def(py::init<const Config&,size_t>())
    .def("step",&Batch::step,py::call_guard<py::gil_scoped_release>())
    .def("step_one",[](Batch& b,size_t i,std::int64_t t,const Feedback& f){return Output(b.governors.at(i).step(t,f));})
    .def("reset",[](Batch& b,size_t i,std::uint64_t s,std::int64_t t,const Feedback& f){return b.governors.at(i).reset(s,t,f);})
    .def("submit",[](Batch& b,size_t i,const Message& m,std::int64_t t){return b.governors.at(i).submit(m,t);})
    .def("stop",[](Batch& b,size_t i){b.governors.at(i).stop();})
    .def("output",[](const Batch& b,size_t i){return Output(b.governors.at(i).output());});
}
