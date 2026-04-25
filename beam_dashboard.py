import sys
import json
import logging
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from scipy.optimize import newton, bisect

# ----------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("beam_dashboard")


# ----------------------------------------------------------------------
# CNT concrete material model
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CNTConcrete:
    """Properties of CNT‑conductive concrete (Hognestad compression, linear tension)."""

    f_c: float          # Compressive strength (Pa)
    eps_c1: float       # Strain at peak stress
    eps_cu: float       # Ultimate compressive strain
    f_t: float          # Tensile strength (Pa)
    eps_tu: float       # Ultimate tensile strain
    base_resistivity: float  # Ohm·m
    gauge_factor: float      # Gauge factor (dimensionless)

    @property
    def E_ci(self) -> float:
        """Initial tangent modulus (slope at origin)."""
        return 2.0 * self.f_c / self.eps_c1

    def stress(self, eps: float) -> float:
        """
        Complete stress-strain law.
        Convention: compression positive, tension negative.
        """
        if eps >= 0.0:          # Compression
            if eps <= self.eps_c1:
                eta = eps / self.eps_c1
                return self.f_c * (2.0 * eta - eta * eta)
            elif eps <= self.eps_cu:
                return self.f_c - 0.15 * self.f_c * (eps - self.eps_c1) / (self.eps_cu - self.eps_c1)
            else:
                return 0.0
        else:                   # Tension
            if eps >= -self.eps_tu:
                return self.E_ci * eps
            else:
                return 0.0

    def strain_from_stress_ascending(self, sigma: float) -> float:
        """
        Stable inverse of the ascending compression branch.
        """
        if sigma < 0 or sigma > self.f_c:
            raise ValueError(f"Stress {sigma:.2e} Pa outside [0, f_c].")
        s = sigma / self.f_c
        eta = s / (1.0 + np.sqrt(1.0 - s))
        return eta * self.eps_c1


# ----------------------------------------------------------------------
# Rectangular beam section – non-linear integration
# ----------------------------------------------------------------------
@dataclass
class RectangularSection:
    """Rectangular section with exact non-linear stress integration."""

    width: float        # m
    height: float       # m
    material: CNTConcrete
    n_layers: int = 500

    def _strain_profile(self, kappa: float, c: float) -> np.ndarray:
        """Returns strain vector over height. y=0 at top, positive downward."""
        y = np.linspace(0.0, self.height, self.n_layers)
        eps = kappa * (c - y)
        return eps

    def _axial_force(self, c: float, kappa: float) -> float:
        """Resultant axial force for given neutral axis depth and curvature."""
        eps = self._strain_profile(kappa, c)
        sigma = np.vectorize(self.material.stress)(eps)
        dy = self.height / (self.n_layers - 1)
        N = np.sum(sigma) * self.width * dy
        return N

    def _internal_moment(self, c: float, kappa: float) -> float:
        """Internal moment about the geometric centre."""
        y = np.linspace(0.0, self.height, self.n_layers)
        eps = self._strain_profile(kappa, c)
        sigma = np.vectorize(self.material.stress)(eps)
        dy = self.height / (self.n_layers - 1)
        lever = y - self.height / 2.0
        M = np.sum(sigma * lever) * self.width * dy
        return M

    def neutral_axis_depth(self, kappa: float) -> float:
        """
        Finds neutral axis depth c (from top fibre) that zeroes axial force.
        """
        def f(c):
            return self._axial_force(c, kappa)

        c_low, c_high = 0.0, self.height * 2.0
        if f(c_low) * f(c_high) > 0:
            if kappa > 0:
                c_high = self.height * 10.0
            else:
                c_low = -self.height * 10.0
        try:
            c_root = bisect(f, c_low, c_high, xtol=1e-12, maxiter=100)
        except ValueError:
            logger.warning("Axial equilibrium not found; forcing c = height.")
            c_root = self.height
        return c_root

    def moment_from_curvature(self, kappa: float) -> Tuple[float, float]:
        """Returns (internal moment, neutral axis depth) for a given curvature."""
        c = self.neutral_axis_depth(kappa)
        M = self._internal_moment(c, kappa)
        return M, c

    def curvature_from_moment(self, M_target: float) -> Tuple[float, float, float]:
        """
        Solves curvature kappa such that M_int(kappa) = M_target.
        Returns (kappa, M_int, c). Raises error if moment cannot be balanced.
        """
        if M_target <= 0.0:
            raise ValueError("Target moment must be positive.")

        def f(kappa):
            M_int, _ = self.moment_from_curvature(kappa)
            return M_int - M_target

        # Initial elastic guess
        I = self.width * self.height**3 / 12.0
        kappa0 = M_target / (self.material.E_ci * I)

        try:
            kappa_root = newton(f, kappa0, tol=1e-10, maxiter=50)
        except RuntimeError:
            logger.warning("Newton did not converge, falling back to bisection.")
            k_low, k_high = kappa0 * 0.1, kappa0 * 10.0
            for _ in range(10):
                if f(k_low) * f(k_high) < 0:
                    break
                k_low *= 0.5
                k_high *= 2.0
            else:
                raise RuntimeError("Could not bracket curvature root.")
            kappa_root = bisect(f, k_low, k_high, xtol=1e-10)

        M_final, c_final = self.moment_from_curvature(kappa_root)
        return kappa_root, M_final, c_final


# ----------------------------------------------------------------------
# Beam dashboard (piezoresistive health monitoring)
# ----------------------------------------------------------------------
@dataclass
class BeamDashboard:
    """Computes mechanical and electrical response of a 3‑point bending beam."""

    section: RectangularSection
    span: float   # m

    def maximum_moment(self, force: float) -> float:
        """Max bending moment at mid-span for a central point load."""
        return force * self.span / 4.0

    def analyze(self, force: float) -> dict:
        """Full analysis for a given force."""
        M_max = self.maximum_moment(force)
        logger.info(f"Force = {force:.1f} N → Max moment = {M_max:.2f} N·m")

        kappa, M_int, c = self.section.curvature_from_moment(M_max)

        # Top fibre strain (compression)
        eps_top = kappa * c
        sigma_top = self.section.material.stress(eps_top)

        # Piezoresistive change (gauge factor applied to top fibre strain)
        deltaR_over_R = self.section.material.gauge_factor * eps_top

        # Reference resistance
        A0 = self.section.width * self.section.height
        R0 = self.section.material.base_resistivity * self.span / A0
        resistance = R0 * (1.0 + deltaR_over_R)

        # Health status based on stress ratio
        ratio = sigma_top / self.section.material.f_c
        if ratio >= 1.0:
            status = "FAILURE"
        elif ratio >= 0.7:
            status = "WARNING"
        elif ratio >= 0.3:
            status = "CAUTION"
        else:
            status = "SAFE"

        return {
            "force_N": force,
            "moment_max_Nm": M_max,
            "curvature": kappa,
            "neutral_axis_depth_m": c,
            "strain_top": eps_top,
            "stress_top_Pa": sigma_top,
            "deltaR_over_R": deltaR_over_R,
            "resistance_ohm": resistance,
            "health_ratio": ratio,
            "health_status": status,
        }


# ----------------------------------------------------------------------
# CLI and configuration
# ----------------------------------------------------------------------
def load_config(path: str) -> dict:
    """Loads configuration from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def build_dashboard(config: dict) -> BeamDashboard:
    """Builds the dashboard object from configuration."""
    mat_cfg = config["material"]
    material = CNTConcrete(
        f_c=mat_cfg["f_c"],
        eps_c1=mat_cfg["eps_c1"],
        eps_cu=mat_cfg["eps_cu"],
        f_t=mat_cfg.get("f_t", 2.0e6),
        eps_tu=mat_cfg.get("eps_tu", 0.0001),
        base_resistivity=mat_cfg["base_resistivity"],
        gauge_factor=mat_cfg["gauge_factor"],
    )

    geom = config["geometry"]
    section = RectangularSection(
        width=geom["width"],
        height=geom["height"],
        material=material,
        n_layers=config.get("integration", {}).get("n_layers", 500),
    )

    return BeamDashboard(section=section, span=geom["span"])


def main():
    parser = argparse.ArgumentParser(
        description="Structural health dashboard – CNT concrete beam in 3‑point bending"
    )
    parser.add_argument("force", type=float, help="Applied force at mid-span (N)")
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="JSON configuration file (default: config.json)",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    parser.add_argument(
        "-o", "--output", help="Output JSON file for results"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    try:
        config = load_config(args.config)
        dashboard = build_dashboard(config)
        results = dashboard.analyze(args.force)
    except Exception as e:
        logger.exception("Error during analysis")
        sys.exit(1)

    # Friendly output
    print("\n" + "=" * 60)
    print("FLEXURAL RESULTS (smart CNT beam)")
    print("-" * 40)
    print(f"Applied force:                   {results['force_N']:.2f} N")
    print(f"Max bending moment:              {results['moment_max_Nm']:.2f} N·m")
    print(f"Curvature:                       {results['curvature']:.6f} rad/m")
    print(f"Neutral axis depth:              {results['neutral_axis_depth_m']:.4f} m")
    print(f"Top fibre strain:                {results['strain_top']:.6f} ({results['strain_top']*1e6:.0f} µε)")
    print(f"Top fibre stress:                {results['stress_top_Pa']/1e6:.2f} MPa")
    print(f"Relative resistance change:      {results['deltaR_over_R']*100:.2f} %")
    print(f"Estimated electrical resistance: {results['resistance_ohm']:.2f} Ω")
    print(f"Health status:                   {results['health_status']} (ratio {results['health_ratio']:.2f})")
    print("=" * 60)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else o)
        logger.info(f"Results saved to {args.output}")


def run_tests():
    """Minimal unit tests."""
    mat = CNTConcrete(
        f_c=40e6, eps_c1=0.0022, eps_cu=0.0038,
        f_t=2.0e6, eps_tu=0.0001,
        base_resistivity=10.0, gauge_factor=-100.0
    )
    assert mat.stress(0.0) == 0.0
    s_peak = mat.stress(mat.eps_c1)
    assert abs(s_peak - mat.f_c) < 1e-9
    assert mat.stress(mat.eps_cu) > 0.0
    assert mat.stress(mat.eps_cu + 0.001) == 0.0

    eps = mat.strain_from_stress_ascending(s_peak * 0.5)
    assert 0 < eps < mat.eps_c1

    # Linear elastic test
    mat_linear = CNTConcrete(
        f_c=1e12, eps_c1=100, eps_cu=200,
        f_t=1e12, eps_tu=100,
        base_resistivity=1.0, gauge_factor=0.0
    )
    section = RectangularSection(0.5, 0.3, mat_linear, n_layers=1000)
    I = section.width * section.height**3 / 12.0
    M_test = 1e3
    kappa_exact = M_test / (mat_linear.E_ci * I)
    kappa_calc, _, _ = section.curvature_from_moment(M_test)
    assert abs(kappa_calc - kappa_exact) < 1e-6 * kappa_exact

    dash = BeamDashboard(section, span=1.0)
    res = dash.analyze(1000.0)
    assert res["moment_max_Nm"] == 250.0
    logger.info("All unit tests passed successfully.")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_tests()
    else:
        main()