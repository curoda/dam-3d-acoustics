import streamlit as st
import pandas as pd
import numpy as np

def main():
    st.title("NxN Pressure Matrix Calculation")

    st.write(
        "Upload an Excel file with the following columns (N rows):\n"
        "`Index, l, m, n, p, q, r, vel x, vel y, vel z, Vx, Vy, Vz`."
    )
    
    # Let the user set the denominator (W*λ), defaulting to 6800
    denominator = st.number_input("Set the denominator (W*λ)", min_value=1, value=6800)

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])
    if uploaded_file is not None:
        # Read the spreadsheet into a DataFrame
        df = pd.read_excel(uploaded_file)
        
        # Determine how many rows (N) we have
        num_points = len(df)
        if num_points < 1:
            st.error("The file must contain at least 1 row of data.")
            return

        # Extract columns as numpy arrays (column names must match exactly)
        l_vals  = df["l"].values
        m_vals  = df["m"].values
        n_vals  = df["n"].values
        p_vals  = df["p"].values
        q_vals  = df["q"].values
        r_vals  = df["r"].values

        vx_vals = df["Vx"].values
        vy_vals = df["Vy"].values
        vz_vals = df["Vz"].values

        velx_vals = df["vel x"].values
        vely_vals = df["vel y"].values
        velz_vals = df["vel z"].values

        # Build the NxN matrix G
        # Rows correspond to (p,q,r) from row i and
        # columns correspond to (l,m,n) from row j
        G = np.empty((num_points, num_points), dtype=np.complex128)
        for i in range(num_points):
            for j in range(num_points):
                exponent = (2.0 * np.pi / denominator) * (
                    p_vals[i] * l_vals[j] + q_vals[i] * m_vals[j] + r_vals[i] * n_vals[j]
                )
                G[i, j] = np.cos(exponent) - 1j * np.sin(exponent)

        # Compute gcc = complex conjugate of G, then take its transpose if needed later.
        gcc = np.conjugate(G)
        gcc_T = gcc.T

        # Display the matrices G and gcc
        st.subheader("Matrix G")
        st.write(pd.DataFrame(G))
        
        st.subheader("Matrix gcc (Complex Conjugate of G)")
        st.write(pd.DataFrame(gcc))

        # Helper function to compute the pressure vector
        def compute_pressure(V_diag_vals, vel_col_vals, label):
            """Compute the pressure vector for the given axis (x, y, or z)."""
            # Create the diagonal matrix from V values
            V_diag = np.diag(V_diag_vals)
            # Ensure the velocity vector is in column format
            vel_vector = vel_col_vals.reshape((num_points, 1))
            # Calculate the pressure: p = G * V_diag * gcc_T * vel_vector
            p_result = G @ V_diag @ gcc_T

            p_real = np.real(p_result).flatten()
            p_imag = np.imag(p_result).flatten()
            p_phase = np.angle(p_result).flatten()

            st.subheader(f"Results for p{label}")
            st.write(f"Real part of p{label}:", p_real)
            st.write(f"Imaginary part of p{label}:", p_imag)

            phase_df = pd.DataFrame({
                "Index": range(1, num_points + 1),
                f"Phase p{label}": p_phase
            })
            st.line_chart(phase_df.set_index("Index"))
            return p_real, p_imag

        # Compute pressure vectors for x, y, and z
        px_real, px_imag = compute_pressure(vx_vals, velx_vals, "x")
        py_real, py_imag = compute_pressure(vy_vals, vely_vals, "y")
        pz_real, pz_imag = compute_pressure(vz_vals, velz_vals, "z")

        # Sum the real parts and imaginary parts of all pressure vectors
        sum_real = px_real + py_real + pz_real
        sum_imag = px_imag + py_imag + pz_imag

        st.subheader("Summation of Real Parts and Imaginary Parts (px + py + pz)")
        result_df = pd.DataFrame({
            "Index": range(1, num_points + 1),
            "Sum Real": sum_real,
            "Sum Imag": sum_imag
        })
        st.dataframe(result_df)
        st.line_chart(result_df.set_index("Index")[["Sum Real", "Sum Imag"]])
    else:
        st.info("Please upload an Excel file to begin.")

if __name__ == "__main__":
    main()
