SELECT
    star_name,
    ROUND(AVG(ra)/1e6, 3)  AS ra_deg,
    ROUND(AVG(dec)/1e6, 3) AS dec_deg,
    COUNT(*)               AS n_obs,
    COUNT(DISTINCT station_name) AS n_cameras,
    ROUND(AVG(cat_mag)/1e6, 3)   AS avg_cat_mag,
    ROUND(AVG(mag)/1e6, 3)       AS avg_obs_mag,
    ROUND((AVG(mag) - AVG(cat_mag))/1e6, 3) AS mag_diff
FROM observation
WHERE star_name IS NOT NULL
GROUP BY star_name
ORDER BY
    n_cameras DESC,
    ABS(AVG(mag) - AVG(cat_mag)) ASC
LIMIT 10;

