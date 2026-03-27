def plotStars(ax, theta, r, names_list=None, ra_deg=None, dec_deg=None, mag=None, color="#66aaff", alpha=0.6, size=None):
    """Scatter-plot stars in polar coordinates."""

    if size is None:
        size = np.clip(4.5 - mag, 0.5, 3.0)


    ax.scatter(theta, r, s=size, zorder=1000, alpha=alpha, color=color)


    if names_list is not None and ra_deg is not None and dec_deg is not None:
        for th, rr, nm, ra, dec in zip(theta, r, names_list, ra_deg, dec_deg):
            label = f"{nm}\nRA={ra:.1f}°\nDec={dec:.1f}°"
            ax.text(th, rr, label,
                    fontsize=7,
                    ha="left", va="bottom",
                    color="black",
                    zorder=2001)

    #ax.scatter(theta, r, s=size, c=color, alpha=alpha)


