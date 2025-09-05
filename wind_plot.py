#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------- 风况配置 ----------------
PROFILES_ALL = {
    "0mps":  {"kind":"const","mag":0.0},
    "5mps":  {"kind":"const","mag":5.0},
    "10mps": {"kind":"const","mag":10.0},
    "12mps": {"kind":"const","mag":12.0},
    "13.5mps":{"kind":"const","mag":13.5},
    "15mps": {"kind":"const","mag":15.0},
    "sinusoidal_0to10mps":{"kind":"sin","mag_mean":5.0,"mag_amp":5.0,"freq_hz":0.33},
    "sinusoidal_0to18mps":{"kind":"sin","mag_mean":9.0,"mag_amp":9.0,"freq_hz":0.25},
    "ou15": {"kind":"ou3d","mean":15.0,"sigma":1.5,"tau":2.0},
    "gustbursts": {"kind":"gust","base":12.0,"amp":6.0,"duration":3.0,"period":12.0},
}

# ---------------- 采样函数 ----------------
def sample_profile(profile, t, dt=0.02, state=None):
    kind = profile["kind"]
    if kind == "const":
        return profile["mag"], state
    elif kind == "sin":
        v = profile["mag_mean"] + profile["mag_amp"]*np.sin(2*np.pi*profile["freq_hz"]*t)
        return v, state
    elif kind == "gust":
        base, amp, dur, per = profile["base"], profile["amp"], profile["duration"], profile["period"]
        in_burst = (t % per) < dur
        return base + (amp if in_burst else 0.0), state
    elif kind == "ou3d":
        mu, sigma, tau = profile["mean"], profile["sigma"], profile["tau"]
        if state is None:
            state = mu
        dW = np.random.normal()
        state += (mu - state) * (dt/tau) + np.sqrt(2*dt/tau)*sigma*dW
        return state, state
    else:
        return 0.0, state

# ---------------- 主绘图 ----------------
def plot_profiles():
    T = 30.0
    dt = 0.02
    t = np.arange(0, T, dt)

    fig, axs = plt.subplots(2, 2, figsize=(12,9))

    # (1) 合并 Const 风，用 colormap + colorbar
    const_profiles = ["0mps","5mps","10mps","12mps","13.5mps","15mps"]
    mags = [PROFILES_ALL[name]["mag"] for name in const_profiles]
    cmap = cm.get_cmap("coolwarm")  # 蓝-红渐变
    norm = plt.Normalize(min(mags), max(mags))

    for name in const_profiles:
        mag = PROFILES_ALL[name]["mag"]
        values = np.ones_like(t) * mag
        axs[0,0].plot(t, values, color=cmap(norm(mag)), label=f"{name} ({mag} m/s)")
    axs[0,0].set_title("Const Winds (0–15 m/s)")
    axs[0,0].set_xlabel("Time [s]")
    axs[0,0].set_ylabel("Wind speed [m/s]")
    axs[0,0].legend()

    # 给定常风图加 colorbar
    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])  # 必须有
    # cbar = fig.colorbar(sm, ax=axs[0,0])
    # cbar.set_label("Wind speed [m/s]")

    # (2) Sin 风
    for name in ["sinusoidal_0to10mps","sinusoidal_0to18mps"]:
        prof = PROFILES_ALL[name]
        values = [sample_profile(prof, ti)[0] for ti in t]
        axs[0,1].plot(t, values, label=name)
    axs[0,1].set_title("Sinusoidal Winds")
    axs[0,1].set_xlabel("Time [s]")
    axs[0,1].set_ylabel("Wind speed [m/s]")
    axs[0,1].legend()

    # (3) Gustbursts
    prof = PROFILES_ALL["gustbursts"]
    gust = [sample_profile(prof, ti)[0] for ti in t]
    axs[1,0].plot(t, gust, label="gustbursts", color="orange")
    axs[1,0].set_title("Gust Bursts")
    axs[1,0].set_xlabel("Time [s]")
    axs[1,0].set_ylabel("Wind speed [m/s]")
    axs[1,0].legend()

    # (4) OU 湍流
    prof = PROFILES_ALL["ou15"]
    ou, state = [], None
    for ti in t:
        v, state = sample_profile(prof, ti, dt, state)
        ou.append(v)
    axs[1,1].plot(t, ou, label="ou15 turbulence", color="green")
    axs[1,1].set_title("OU Turbulence")
    axs[1,1].set_xlabel("Time [s]")
    axs[1,1].set_ylabel("Wind speed [m/s]")
    axs[1,1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_profiles()
