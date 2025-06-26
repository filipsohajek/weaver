#pragma once

#include "weaver/systems.h"
#include "weaver/types.h"

namespace weaver {
struct TimeOfWeek {
  static const constexpr f64 WEEK_SECONDS = 604800;
  GNSSSystem sys;
  f64 tow;

  f64 operator-(const TimeOfWeek& rhs) const {
    assert(sys == rhs.sys);
    f64 tow_diff = tow - rhs.tow;
    if (tow_diff > WEEK_SECONDS / 2) {
      tow_diff -= WEEK_SECONDS;
    } else if (tow_diff < -WEEK_SECONDS / 2) {
      tow_diff += WEEK_SECONDS;
    }

    return tow_diff;
  }

  TimeOfWeek& operator+=(f64 tow_inc) {
    tow = std::fmod(tow + tow_inc, WEEK_SECONDS);
    return *this;
  }

  TimeOfWeek operator+(f64 tow_inc) {
    TimeOfWeek new_time = *this;
    new_time += tow_inc;
    return new_time;
  }
};

struct WGS84Position {
  static const constexpr f64 A = 6378137.0;
  static const constexpr f64 B = 6356752.3142;

  f64 x;
  f64 y;
  f64 z;

  [[nodiscard]] Eigen::Vector3d vector() const { return {x, y, z}; }

  [[nodiscard]] std::tuple<f64, f64, f64> lat_lon_height() const {
    f64 e = sqrt((A * A - B * B) / (A * A));
    f64 e_prime = sqrt((A * A - B * B) / (B * B));

    f64 p = sqrt(x * x + y * y);
    f64 F = 54 * B * B * z * z;
    f64 G = p * p + (1 - e * e) * z * z - e * e * (A * A - B * B);
    f64 c = (pow(e, 4) * F * p * p) / pow(G, 3);
    f64 s = cbrt(1 + c + sqrt(c * c + 2 * c));
    f64 k = s + 1 + 1 / s;
    f64 P = F / (3 * k * k * G * G);
    f64 Q = sqrt(1 + 2 * pow(e, 4) * P);
    f64 r_0 = (-P * e * e * p) / (1 + Q) +
              sqrt(0.5 * A * A * (1 + 1 / Q) - (P * (1 - e * e) * z * z) / (Q * (1 + Q)) -
                   0.5 * P * p * p);
    f64 U = sqrt(pow(p - e * e * r_0, 2) + z * z);
    f64 V = sqrt(pow(p - e * e * r_0, 2) + (1 - e * e) * z * z);
    f64 z_0 = (B * B * z) / (A * V);

    f64 h = U * (1 - (B * B) / (A * V));
    f64 lat = atan((z + e_prime * e_prime * z_0) / p);
    f64 lon = atan2(y, x);

    return std::make_tuple(lat, lon, h);
  }

  [[nodiscard]] std::tuple<f64, f64, f64> enu_coordinates(WGS84Position ref) {
    f64 delta_x = x - ref.x;
    f64 delta_y = y - ref.y;
    f64 delta_z = z - ref.z;

    auto [ref_lat, ref_lon, _] = ref.lat_lon_height();
    f64 east = -delta_x * sin(ref_lon) + delta_y * cos(ref_lon);
    f64 north = -delta_x * sin(ref_lat) * cos(ref_lon) - delta_y * sin(ref_lat) * sin(ref_lon) +
                delta_z * cos(ref_lat);
    f64 up = delta_x * cos(ref_lat) * cos(ref_lon) + delta_y * cos(ref_lat) * sin(ref_lon) +
             delta_z * sin(ref_lat);

    return std::make_tuple(east, north, up);
  }
};

struct Ephemeris {
  static const constexpr f64 GRAV_CONST_WGS84 = 3.986005e14;
  static const constexpr f64 LIGHT_SPEED = 299792458;
  static const constexpr f64 EARTH_ROT_RATE_WGS84 = 7.2921151467e-5;

  GNSSSystem sys;

  f64 mean_anomaly;
  f64 mean_motion_diff;
  f64 eccentricity;
  f64 semmaj_axis_sqrt;
  f64 asc_node_lon;
  f64 inc_angle;
  f64 arg_perigee;
  f64 right_asc_rate;
  f64 inc_angle_rate;

  f64 arg_lat_cos_corr;
  f64 arg_lat_sin_corr;
  f64 orbit_r_cos_corr;
  f64 orbit_r_sin_corr;
  f64 inc_angle_cos_corr;
  f64 inc_angle_sin_corr;

  TimeOfWeek ref_tow;
  u16 week_nr;
  u16 issue;

  struct {
    f64 offset;
    f64 drift;
    f64 frequency_drift;
    f64 group_delay;

    TimeOfWeek ref_tow;
    u32 issue;
  } clock_params;

  [[nodiscard]] f64 ecc_anomaly(TimeOfWeek sys_time, u8 n_iter = 5) const;
  [[nodiscard]] WGS84Position position(TimeOfWeek sys_time) const {
    f64 time_diff = sys_time - ref_tow;
    f64 ecc_anom = ecc_anomaly(sys_time);
    f64 true_anomaly =
        2 * std::atan(std::sqrt((1 + eccentricity) / (1 - eccentricity)) * std::tan(ecc_anom / 2));
    f64 arg_latitude = true_anomaly + arg_perigee;
    f64 radius = semmaj_axis_sqrt * semmaj_axis_sqrt * (1 - eccentricity * std::cos(ecc_anom));
    f64 inclination = inc_angle + inc_angle_rate * time_diff;

    radius += orbit_r_sin_corr * std::sin(2 * arg_latitude) +
              orbit_r_cos_corr * std::cos(2 * arg_latitude);
    inclination += inc_angle_sin_corr * std::sin(2 * arg_latitude) +
                   inc_angle_cos_corr * std::cos(2 * arg_latitude);
    arg_latitude += arg_lat_sin_corr * std::sin(2 * arg_latitude) +
                    arg_lat_cos_corr * std::cos(2 * arg_latitude);

    f64 orb_x = radius * std::cos(arg_latitude);
    f64 orb_y = radius * std::sin(arg_latitude);
    f64 lon_asc_node_corr = asc_node_lon + (right_asc_rate - EARTH_ROT_RATE_WGS84) * time_diff -
                            EARTH_ROT_RATE_WGS84 * ref_tow.tow;

    return WGS84Position{.x = orb_x * std::cos(lon_asc_node_corr) -
                              orb_y * std::cos(inclination) * std::sin(lon_asc_node_corr),
                         .y = orb_x * std::sin(lon_asc_node_corr) +
                              orb_y * std::cos(inclination) * std::cos(lon_asc_node_corr),
                         .z = orb_y * std::sin(inclination)};
  }

  [[nodiscard]] f64 clock_corr(TimeOfWeek sys_time, bool is_single_freq = true) {
    const f64 F = -2 * std::sqrt(GRAV_CONST_WGS84) / std::pow(LIGHT_SPEED, 2);
    f64 rel_clock_corr = F * eccentricity * semmaj_axis_sqrt * std::sin(ecc_anomaly(sys_time));

    f64 clock_diff = sys_time - ref_tow;
    f64 sv_clock_corr = clock_params.offset + clock_diff * clock_params.drift +
                        clock_diff * clock_diff * clock_params.frequency_drift;

    return rel_clock_corr + sv_clock_corr + (is_single_freq ? clock_params.group_delay : 0);
  }
};

struct Measurement {
  SignalID signal;
  TimeOfWeek tow;
  std::optional<f64> code_pseudorange;
  std::optional<f64> carrier_pseudorange;
};
};  // namespace weaver
