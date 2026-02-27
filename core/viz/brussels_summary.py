import streamlit as st

def render_brussels_summary(mode: str, gdf_filtered, filters: dict):
    """
    Render a mode-dependent summary block below the map.
    Keep this function UI-only: it receives already-filtered data.
    """
    st.markdown("### Summary")

    total_features = len(gdf_filtered)
    st.metric("Displayed features", total_features)

    # Show active filters
    with st.expander("Active filters", expanded=False):
        st.json({"mode": mode, "filters": filters})

    # Mode-specific summary (prototype)
    if mode == "Segment":
        cols = st.columns(3)
        with cols[0]:
            st.metric("Selected segment names", len(filters.get("segment_names", [])))
        with cols[1]:
            st.metric("Selected bus IDs", len(filters.get("bus_ids", [])))
        with cols[2]:
            # Example: how many unique lines/variants visible (if present)
            if "ligne" in gdf_filtered.columns:
                st.metric("Unique lines", gdf_filtered["ligne"].nunique(dropna=True))
            else:
                st.metric("Unique lines", "N/A")

        # Optional: show a small table if columns exist
        preview_cols = [c for c in ["ligne", "variante", "__speed_str"] if c in gdf_filtered.columns]
        if preview_cols:
            st.dataframe(
                gdf_filtered[preview_cols].head(30),
                use_container_width=True,
            )

    elif mode == "Street":
        cols = st.columns(2)
        with cols[0]:
            st.metric("Selected streets", len(filters.get("streets", [])))
        with cols[1]:
            if "ligne" in gdf_filtered.columns:
                st.metric("Unique lines", gdf_filtered["ligne"].nunique(dropna=True))
            else:
                st.metric("Unique lines", "N/A")

        preview_cols = [c for c in ["ligne", "variante"] if c in gdf_filtered.columns]
        if preview_cols:
            st.dataframe(
                gdf_filtered[preview_cols].head(30),
                use_container_width=True,
            )
