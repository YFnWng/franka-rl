#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace franka_governor {
constexpr std::int64_t tick_ns = 1000000;
constexpr double dt = 0.001;
using Vec = std::array<double, 7>;
using Raw = std::array<float, 7>;
enum class Status { Disarmed, Running, Stopping, Terminal, Fault };
enum class Reason { None, RequestedStop, InvalidAction, Sequence, Session, FutureTime,
  StaleAction, StaleObservation, StateInvalid, Timing, Tracking, Infeasible, Intervention };
struct Config {
  Vec lower{}, upper{}, margin{}, max_velocity{}, max_acceleration{}, max_jerk{};
  Vec envelope_velocity{}, envelope_offset{}, envelope_deceleration{};
  Vec default_position{}, tracking_error{}, desired_error{}, desired_velocity_error{}, desired_acceleration_error{};
  double max_segment_distance{}, max_projection{};
  int horizon_ticks{}, max_blocked_ticks{}, max_projected_ticks{};
  std::int64_t action_timeout_ns{}, observation_timeout_ns{}, tick_tolerance_ns{};
  void validate() const {
    if (horizon_ticks < 2 || horizon_ticks > 2000 || max_blocked_ticks < 1 ||
        max_projected_ticks < 1 || !std::isfinite(max_segment_distance) || max_segment_distance <= 0 ||
        !std::isfinite(max_projection) || max_projection < 0 || action_timeout_ns < tick_ns ||
        observation_timeout_ns < 0 || tick_tolerance_ns < 0 || tick_tolerance_ns >= tick_ns)
      throw std::invalid_argument("invalid required governor scalar configuration");
    for (int i=0;i<7;++i) {
      for (double x: {lower[i],upper[i],margin[i],max_velocity[i],max_acceleration[i],max_jerk[i],
                     envelope_velocity[i],envelope_offset[i],envelope_deceleration[i],
                     default_position[i],tracking_error[i],desired_error[i],desired_velocity_error[i],desired_acceleration_error[i]})
        if (!std::isfinite(x)) throw std::invalid_argument("nonfinite configuration");
      if (margin[i]<=0 || lower[i]+margin[i]>=upper[i]-margin[i] || max_velocity[i]<=0 ||
          max_acceleration[i]<=0 || max_acceleration[i]>9.999 || max_jerk[i]<=0 || max_jerk[i]>4999.999 || envelope_velocity[i]<=0 ||
          envelope_offset[i]<0 || envelope_deceleration[i]<=0 || tracking_error[i]<=0 || desired_error[i]<=0 ||
          desired_velocity_error[i]<=0 || desired_acceleration_error[i]<=0 ||
          default_position[i]<lower[i]+margin[i] || default_position[i]>upper[i]-margin[i])
        throw std::invalid_argument("invalid joint limits, defaults or tolerances");
    }
  }
};
struct Feedback {
  Vec q{}, dq{}, desired_q{}, desired_dq{}, desired_ddq{};
  std::int64_t observed_ns{};
  std::uint64_t session{};
  bool healthy{true};
};
struct Message {
  Raw action{};
  std::uint64_t sequence{}, session{};
  std::int64_t observation_ns{}, completed_ns{};
};
struct Output {
  Vec q{}, dq{}, ddq{}, mapped_target{}, projected_target{};
  Raw previous_raw{};
  Status status{Status::Disarmed};
  Reason reason{Reason::None};
  std::uint64_t accepted_sequence{};
  bool command_valid{false}, replanning_blocked{false};
  double projection{};
};

// Quintic Bezier segments. Convex hulls of derivative control points bound the
// entire continuous trajectory, not merely sampled endpoints. No dynamic memory
// in reset/submit/step/stop. Each accepted segment has zero terminal velocity and
// acceleration and provides the cached stopping continuation.
class Governor {
 public:
  explicit Governor(const Config& config): c_(config) { c_.validate(); }
  const Output& output() const noexcept { return o_; }
  const Config& config() const noexcept { return c_; }
  bool reset(std::uint64_t session, std::int64_t now, const Feedback& f) noexcept {
    if (session==0 || session<=session_ || now<0) { fault(Reason::Session); return false; }
    // A new session is explicit. Failure cannot leave an old command usable.
    session_=session; o_=Output{}; o_.status=Status::Fault; o_.reason=Reason::Infeasible;
    last_now_=now; last_observed_=f.observed_ns; last_action_=now;
    pending_=false; blocked_=projected_=0; planned_sequence_=0;
    if (f.session!=session || !finite(f) || !f.healthy || f.observed_ns<0 ||
        f.observed_ns>now || now-f.observed_ns>c_.observation_timeout_ns) {
      o_.reason=Reason::StateInvalid; return false;
    }
    o_.q=f.desired_q; o_.dq=f.desired_dq; o_.ddq=f.desired_ddq;
    if (!feedback_ok(f)) { o_.reason=Reason::Tracking; return false; }
    Vec endpoint{};
    for(int i=0;i<7;++i) endpoint[i]=std::clamp(o_.q[i]+o_.dq[i]*duration()/2,
                                              c_.lower[i]+c_.margin[i],c_.upper[i]-c_.margin[i]);
    if (!make_segment(endpoint)) return false;
    target_=endpoint; o_.mapped_target=endpoint; o_.projected_target=endpoint;
    o_.status=Status::Running; o_.reason=Reason::None; o_.command_valid=true;
    return true;
  }
  bool submit(const Message& m, std::int64_t now) noexcept {
    if (o_.status!=Status::Running) return false;
    Reason r=Reason::None;
    if(m.session!=session_) r=Reason::Session;
    else if(pending_ || m.sequence!=o_.accepted_sequence+1) r=Reason::Sequence;
    else if(now<last_now_ || m.completed_ns>now || m.observation_ns>m.completed_ns) r=Reason::FutureTime;
    else if(m.observation_ns<0 || now-m.observation_ns>c_.observation_timeout_ns) r=Reason::StaleObservation;
    else if(m.completed_ns<0 || now-m.completed_ns>c_.action_timeout_ns) r=Reason::StaleAction;
    else for(float x:m.action) if(!std::isfinite(x)) r=Reason::InvalidAction;
    if(r!=Reason::None) { stop(r); return false; }
    pending_message_=m; pending_=true; return true;
  }
  void stop(Reason r=Reason::RequestedStop) noexcept {
    if(o_.status==Status::Running) { o_.status=Status::Stopping; o_.reason=r; pending_=false; }
  }
  const Output& step(std::int64_t now, const Feedback& f) noexcept {
    if(o_.status==Status::Disarmed || o_.status==Status::Fault) return o_;
    if(now<=last_now_ || std::abs((now-last_now_)-tick_ns)>c_.tick_tolerance_ns) return fault(Reason::Timing);
    if(f.session!=session_) return fault(Reason::Session);
    if(!f.healthy || !finite(f) || f.observed_ns<=last_observed_ || f.observed_ns>now ||
       now-f.observed_ns>c_.observation_timeout_ns) return fault(Reason::StateInvalid);
    if(!feedback_ok(f)) return fault(Reason::Tracking);
    last_now_=now; last_observed_=f.observed_ns;
    if(o_.status==Status::Terminal) return o_;
    // Advance cached, already certified trajectory before accepting a retarget.
    if(elapsed_<c_.horizon_ticks) ++elapsed_;
    evaluate();
    if(o_.status==Status::Running && pending_) {
      auto m=pending_message_; pending_=false;
      if(now-m.observation_ns>c_.observation_timeout_ns || now<m.completed_ns)
        stop(Reason::StaleObservation);
      else if(now-m.completed_ns>c_.action_timeout_ns) stop(Reason::StaleAction);
      else {
        o_.projection=0;
        for(int i=0;i<7;++i) {
          // Explicit float32 action contract; CMake disables multiply-add fusion.
          const float scaled=0.5f*m.action[i];
          const float mapped=static_cast<float>(c_.default_position[i])+scaled;
          o_.mapped_target[i]=mapped;
          target_[i]=std::clamp(double(mapped),c_.lower[i]+c_.margin[i],c_.upper[i]-c_.margin[i]);
          o_.projection=std::max(o_.projection,std::abs(double(mapped)-target_[i]));
        }
        o_.projected_target=target_; o_.previous_raw=m.action; o_.accepted_sequence=m.sequence;
        last_action_=m.completed_ns;
        if(o_.projection>c_.max_projection) stop(Reason::Intervention);
      }
    }
    if(o_.status==Status::Running && now-last_action_>c_.action_timeout_ns) stop(Reason::StaleAction);
    if(o_.status==Status::Running) {
      projected_=o_.projection>0 ? projected_+1:0;
      if(projected_>=c_.max_projected_ticks) stop(Reason::Intervention);
    }
    if(o_.status==Status::Running) {
      // Replan at policy arrival or completion only, not every tick (which would
      // continually postpone the terminal state). Rejected proposals retain the
      // previous certified continuation. Retry at the next target or at rest.
      if(o_.accepted_sequence!=planned_sequence_ || elapsed_==c_.horizon_ticks) {
        bool found=false;
        for(int attempt=0;attempt<12 && !found;++attempt) {
          Vec end{}; const double scale=std::ldexp(1.0,-attempt);
          for(int i=0;i<7;++i) end[i]=o_.q[i]+scale*std::clamp(target_[i]-o_.q[i],
              -c_.max_segment_distance,c_.max_segment_distance);
          found=make_segment(end);
        }
        planned_sequence_=o_.accepted_sequence; o_.replanning_blocked=!found;
      }
      blocked_=o_.replanning_blocked?blocked_+1:0;
      if(blocked_>=c_.max_blocked_ticks) stop(Reason::Intervention);
    }
    if(o_.status==Status::Stopping && elapsed_==c_.horizon_ticks) o_.status=Status::Terminal;
    return o_;
  }
 private:
  Config c_; Output o_{}; Vec target_{};
  std::array<std::array<double,6>,7> p_{};
  int elapsed_{},blocked_{},projected_{};
  std::uint64_t session_{},planned_sequence_{};
  std::int64_t last_now_{},last_action_{},last_observed_{};
  bool pending_{}; Message pending_message_{};
  double duration() const noexcept { return c_.horizon_ticks*dt; }
  static bool finite(const Feedback& f) noexcept {
    for(int i=0;i<7;++i) for(double x:{f.q[i],f.dq[i],f.desired_q[i],f.desired_dq[i],f.desired_ddq[i]})
      if(!std::isfinite(x)) return false;
    return true;
  }
  bool feedback_ok(const Feedback& f) const noexcept {
    for(int i=0;i<7;++i) {
      if(f.q[i]<c_.lower[i]+c_.margin[i] || f.q[i]>c_.upper[i]-c_.margin[i] ||
         std::abs(f.q[i]-o_.q[i])>c_.tracking_error[i] ||
         std::abs(f.desired_q[i]-o_.q[i])>c_.desired_error[i] ||
         std::abs(f.desired_dq[i]-o_.dq[i])>c_.desired_velocity_error[i] ||
         std::abs(f.desired_ddq[i]-o_.ddq[i])>c_.desired_acceleration_error[i] ||
         std::abs(f.dq[i])>c_.max_velocity[i] || std::abs(f.desired_dq[i])>c_.max_velocity[i] ||
         std::abs(f.desired_ddq[i])>c_.max_acceleration[i]) return false;
    }
    return true;
  }
  const Output& fault(Reason r) noexcept { o_.status=Status::Fault;o_.reason=r;o_.command_valid=false;pending_=false;return o_; }
  bool make_segment(const Vec& endpoint) noexcept {
    std::array<std::array<double,6>,7> candidate{};
    const double T=duration();
    for(int i=0;i<7;++i) {
      auto& b=candidate[i];
      b={o_.q[i],o_.q[i]+o_.dq[i]*T/5,
         o_.q[i]+2*o_.dq[i]*T/5+o_.ddq[i]*T*T/20,endpoint[i],endpoint[i],endpoint[i]};
      const auto mm=std::minmax_element(b.begin(),b.end());
      for(double value:b) if(!std::isfinite(value)) return false;
      if(*mm.first<c_.lower[i]+c_.margin[i] || *mm.second>c_.upper[i]-c_.margin[i] ||
         *mm.first<o_.q[i]-c_.max_segment_distance || *mm.second>o_.q[i]+c_.max_segment_distance) return false;
      // Monotone envelope: the minimum upper velocity is at largest position;
      // the maximum lower velocity is at smallest position over this hull.
      const double hi=std::min(c_.max_velocity[i],std::min(c_.envelope_velocity[i],std::max(0.,
          -c_.envelope_offset[i]+std::sqrt(std::max(0.,2*c_.envelope_deceleration[i]*(c_.upper[i]-*mm.second)))))-.001);
      const double lo=std::max(-c_.max_velocity[i],std::max(-c_.envelope_velocity[i],std::min(0.,
          c_.envelope_offset[i]-std::sqrt(std::max(0.,2*c_.envelope_deceleration[i]*(*mm.first-c_.lower[i])))))+.001);
      std::array<double,5> v{};std::array<double,4>a{};
      for(int k=0;k<5;++k) { v[k]=5*(b[k+1]-b[k])/T; if(v[k]<lo || v[k]>hi) return false; }
      for(int k=0;k<4;++k) { a[k]=4*(v[k+1]-v[k])/T; if(std::abs(a[k])>c_.max_acceleration[i]) return false; }
      for(int k=0;k<3;++k) if(std::abs(3*(a[k+1]-a[k])/T)>c_.max_jerk[i]) return false;
    }
    p_=candidate;elapsed_=0;return true;
  }
  template<size_t N> static double bezier(std::array<double,N> b,double u) noexcept {
    for(size_t n=N-1;n>0;--n) for(size_t i=0;i<n;++i) b[i]=std::clamp((1-u)*b[i]+u*b[i+1],std::min(b[i],b[i+1]),std::max(b[i],b[i+1]));
    return b[0];
  }
  void evaluate() noexcept {
    const double u=double(elapsed_)/c_.horizon_ticks,T=duration();
    for(int i=0;i<7;++i) {
      std::array<double,5> v{};std::array<double,4>a{};
      for(int k=0;k<5;++k) v[k]=5*(p_[i][k+1]-p_[i][k])/T;
      for(int k=0;k<4;++k) a[k]=4*(v[k+1]-v[k])/T;
      o_.q[i]=bezier(p_[i],u);o_.dq[i]=bezier(v,u);o_.ddq[i]=bezier(a,u);
    }
  }
};
} // namespace franka_governor
