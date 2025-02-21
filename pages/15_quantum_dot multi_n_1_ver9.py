# %%
import streamlit as st
from qutip import (basis, tensor, Bloch, qeye, sigmax, sigmay, sigmaz, sigmam,
                    Qobj, expect, sesolve, mesolve)
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyperclip
from plotly.subplots import make_subplots
import plotly.express as px
import openai

# Initialize the OpenAI client (if needed)
client = openai.OpenAI(api_key="<key>")

# Set the page layout to wide
st.set_page_config(layout="wide")

# %% Helper Functions

def embed_operator(op, sys_idx, total_spins):
    """Embed an operator 'op' into a system of 'total_spins' spins (spin-1/2)."""
    op_list = [op if i == sys_idx else qeye(2) for i in range(total_spins)]
    return tensor(op_list)

def quadrupole_term(q_val, eta_val, sx, sy, sz):
    """
    Return the quadrupole term operator for a single nuclear spin.
    For spin I>=1, the term is proportional to 3I_z^2 - I(I+1) and an asymmetry part.
    Here, we use a simplified version appropriate for spin-1/2 (which effectively yields zero)
    but if extended to higher spins, the formula can be adjusted.
    """
    iz2 = sz * sz
    ix2 = sx * sx
    iy2 = sy * sy
    eye = qeye(2)
    return (q_val / 6.0) * (3*iz2 - (3.0/4.0)*eye + eta_val*(ix2 - iy2))

@st.cache_resource
def create_total_hamiltonian_with_rf(
    num_nuclei,
    Bz=1.0,
    gamma_e=28.0,
    J_dd=0.1,
    hbar=1.0,
    # RF-drive parameters
    nu_rf=1.0,      # RF frequency
    amp_rf=0.1,     # RF amplitude
    gamma_n=1.0,
    q=0.5,
    eta=0.2,
    A_hf=0.5
):
    """
    Build a time-dependent Hamiltonian for a quantum dot system with
    num_nuclei nuclear spins (spin-1/2) and one electron spin.
    Includes nuclear and electron Zeeman, nuclear quadrupole, nuclear dipole–dipole,
    hyperfine interactions, and a time-dependent RF drive on the nuclei.
    """
    max_nuclei = 20
    if num_nuclei > max_nuclei:
        raise ValueError(f"Only up to {max_nuclei} nuclei are supported.")

    # Define parameter lists (could be made nucleus-dependent)
    gamma_n_list = [gamma_n] * max_nuclei
    q_list       = [q] * max_nuclei
    eta_list     = [eta] * max_nuclei
    A_hf_list    = [A_hf] * max_nuclei

    # Single-spin operators for spin-1/2 (I = σ/2)
    sx = sigmax() / 2.0
    sy = sigmay() / 2.0
    sz = sigmaz() / 2.0

    # Prepare nuclear spin operators for each nucleus (first num_nuclei indices)
    ops_n = []
    for i in range(num_nuclei):
        ops_n.append({
            'x': embed_operator(sx, i, num_nuclei + 1),
            'y': embed_operator(sy, i, num_nuclei + 1),
            'z': embed_operator(sz, i, num_nuclei + 1),
        })

    # Electron spin operators at the last index (index num_nuclei)
    ops_e = {
        'x': embed_operator(sx, num_nuclei, num_nuclei + 1),
        'y': embed_operator(sy, num_nuclei, num_nuclei + 1),
        'z': embed_operator(sz, num_nuclei, num_nuclei + 1),
    }

    # Static Hamiltonian terms
    # (a) Nuclear Zeeman: -hbar * gamma_n * Bz * I_z
    H_nuc_zeeman = sum(-hbar * gamma_n_list[i] * Bz * ops_n[i]['z'] for i in range(num_nuclei))

    # (b) Nuclear Quadrupole (if nuclei have I>=1, this term becomes non-zero)
    H_nuc_quad = sum(embed_operator(quadrupole_term(q_list[i], eta_list[i], sx, sy, sz), i, num_nuclei + 1)
                     for i in range(num_nuclei))

    # (c) Nuclear dipole-dipole coupling among pairs
    H_nuc_dd = 0
    for i in range(num_nuclei):
        for j in range(i+1, num_nuclei):
            H_nuc_dd += J_dd * (ops_n[i]['x']*ops_n[j]['x'] +
                                 ops_n[i]['y']*ops_n[j]['y'] +
                                 ops_n[i]['z']*ops_n[j]['z'])

    # (d) Electron Zeeman: -hbar * gamma_e * Bz * S_z
    H_e_zeeman = -hbar * gamma_e * Bz * ops_e['z']

    # (e) Hyperfine coupling: A_hf * (I . S)
    H_hf = sum(A_hf_list[i] * (ops_n[i]['x']*ops_e['x'] +
                               ops_n[i]['y']*ops_e['y'] +
                               ops_n[i]['z']*ops_e['z'])
               for i in range(num_nuclei))

    # Total static Hamiltonian
    H_static = H_nuc_zeeman + H_nuc_quad + H_nuc_dd + H_e_zeeman + H_hf

    # Time-dependent RF drive on both nuclear spins and electron:
    # H_rf(t) = -hbar * [ cos(2π nu_rf t)*(gamma_n*amp_rf*I_tot_x + gamma_e*amp_rf*S_x) 
    #                     - sin(2π nu_rf t)*(gamma_n*amp_rf*I_tot_y + gamma_e*amp_rf*S_y) ]
    I_tot_x = sum(ops_n[i]['x'] for i in range(num_nuclei))
    I_tot_y = sum(ops_n[i]['y'] for i in range(num_nuclei))

    def coeff_x(t, args):
        return -hbar * np.cos(2.0 * np.pi * args['nu_rf'] * t)
    def coeff_y(t, args):
        return +hbar * np.sin(2.0 * np.pi * args['nu_rf'] * t)

    H_time_dep = [
        H_static,
        [gamma_n * amp_rf * I_tot_x + gamma_e * amp_rf * ops_e['x'], coeff_x],  # Scale by respective gamma
        [gamma_n * amp_rf * I_tot_y + gamma_e * amp_rf * ops_e['y'], coeff_y]   # Scale by respective gamma
    ]

    return H_time_dep, H_static, H_e_zeeman, H_nuc_zeeman

def extract_subsystems_from_eigenstate(H_static, state_idx):
    """
    Diagonalize H_static and return partial-trace density matrices for each nucleus
    and the electron, along with the energy and full eigenstate vector.
    """
    dim = H_static.shape[0]
    total_spins = int(np.log2(dim))
    if 2**total_spins != dim:
        raise ValueError(f"Dimension {dim} is not 2^N for some integer N.")

    num_nuclei = total_spins - 1

    eigenvals, eigenstates = H_static.eigenstates()
    idx_sort = np.argsort(eigenvals)
    eigenvals_sorted = [eigenvals[i] for i in idx_sort]
    eigenstates_sorted = [eigenstates[i] for i in idx_sort]

    psi = eigenstates_sorted[state_idx]
    energy = eigenvals_sorted[state_idx]

    nuclear_dms = [psi.ptrace(i) for i in range(num_nuclei)]
    electron_dm = psi.ptrace(num_nuclei)
    return nuclear_dms, electron_dm, energy, psi

def simulate_time_evolution(H, psi0, tlist, e_ops, args, include_decoherence=False, collapse_ops=None):
    """
    Simulate the time evolution of the system.
    If include_decoherence is False, use sesolve (unitary evolution).
    If True, use mesolve with provided collapse operators.
    """
    if include_decoherence:
        result = mesolve(H, psi0, tlist, c_ops=collapse_ops, e_ops=e_ops, args=args, options={"store_states": True})
    else:
        result = sesolve(H, psi0, tlist, e_ops=e_ops, args=args, options={"store_states": True})
    return result

def create_spin_bloch(dm, title=""):
    """Generate a Bloch sphere plot from a density matrix."""
    b = Bloch()
    b.add_states(dm)
    b.render()
    if title:
        b.fig.suptitle(title)
    b.fig.tight_layout()
    return b.fig

def evolve_and_plot_time_dep(
    H,
    state_index=0,          
    t_max=10.0,
    num_steps=100,
    observable_name=None,   
    times_for_bloch=None,   
    args=None,
    include_decoherence=False,
    collapse_ops=None,
    show_bloch_spheres=False
):
    """
    Evolve an initial state under a (time-dependent) Hamiltonian H.
    Optionally include decoherence by switching from sesolve to mesolve.
    Also plots observables and Bloch spheres for selected times.
    """
    if isinstance(H, list):
        static_part = H[0]
        if not isinstance(static_part, Qobj):
            raise ValueError("H[0] is not a Qobj, can't diagonalize.")
        dim = static_part.shape[0]
    else:
        static_part = H
        dim = H.shape[0]

    total_spins = int(np.log2(dim))
    if 2**total_spins != dim:
        raise ValueError("H dimension not 2^N => not spin-1/2 system?")

    # Construct initial state based on eigenstates of H_static.
    def construct_initial_state(H_static, st_index):
        eigvals, eigstates = H_static.eigenstates()
        idx_sort = np.argsort(eigvals)
        eigvals_sorted = [eigvals[i] for i in idx_sort]
        eigstates_sorted = [eigstates[i] for i in idx_sort]
        max_idx = len(eigstates_sorted) - 1
        floor_idx = int(np.floor(st_index))
        frac_part = st_index - floor_idx
        if floor_idx < 0 or floor_idx > max_idx:
            raise ValueError(f"Requested eigenstate {floor_idx} out of range [0..{max_idx}].")
        if abs(frac_part) < 1e-9:
            return eigstates_sorted[floor_idx]
        if abs(frac_part - 0.5) < 1e-9:
            i1 = floor_idx
            i2 = i1 + 1
            if i2 > max_idx:
                raise ValueError(f"Cannot form superposition: second index {i2} out of range.")
            psi0 = (eigstates_sorted[i1] + eigstates_sorted[i2]).unit()
            return psi0
        raise ValueError(f"state_index must be integer or half-integer. Got {st_index}.")

    psi0 = construct_initial_state(static_part, state_index)

    # Define the Pauli-Z operator for spin-1/2 (scaled by 1/2)
    sz_12 = sigmaz() / 2

    def embed(op, spin_idx):
        op_list = []
        for i in range(total_spins):
            op_list.append(op if i == spin_idx else qeye(2))
        return tensor(op_list)

    electron_idx = total_spins - 1

    def build_observable(obs_name):
        if obs_name is None:
            return None
        obs_name = obs_name.lower()
        if obs_name == "system":
            obs_ = 0
            for i in range(total_spins):
                obs_ += embed(sz_12, i)
            return obs_
        if obs_name == "electron_only":
            return embed(sz_12, electron_idx)
        if obs_name.startswith("nuclear_"):
            try:
                nuc_idx = int(obs_name.replace("nuclear_", "")) - 1
            except:
                raise ValueError(f"Invalid nuclear label: {obs_name}")
            if nuc_idx < 0 or nuc_idx >= electron_idx:
                raise ValueError(f"System has {electron_idx} nuclear spins, but requested {obs_name}.")
            return embed(sz_12, nuc_idx)
        if obs_name == "multi_view":
            obs_system = build_observable("system")
            obs_electron = build_observable("electron_only")
            obs_nuclear_avg = sum(build_observable(f"nuclear_{i+1}") for i in range(electron_idx)) / electron_idx
            return obs_system, obs_electron, obs_nuclear_avg
        return None

    obs = build_observable(observable_name)

    times = np.linspace(0, t_max, num_steps)
    if args is None:
        args = {}

    if obs is not None:
        if observable_name == "multi_view":
            e_ops = list(obs)
        else:
            e_ops = [obs]
    else:
        e_ops = []

    # Optionally include decoherence by using collapse operators (if provided)
    result = simulate_time_evolution(
        H, psi0, times, e_ops, args,
        include_decoherence=include_decoherence,
        collapse_ops=collapse_ops
    )

    # Plotting the observable evolution using Plotly animation.
    if obs is not None:
        if observable_name == "multi_view":
            exp_vals_system, exp_vals_electron, exp_vals_nuclear_avg = result.expect
            fig = make_subplots(rows=3, cols=1,
                                subplot_titles=('System', 'Electron', 'Nuclear'))
            fig.add_trace(
                go.Scatter(x=[times[0]], y=[exp_vals_system[0]], mode='lines', name='System'),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=[times[0]], y=[exp_vals_electron[0]], mode='lines', name='Electron'),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=[times[0]], y=[exp_vals_nuclear_avg[0]], mode='lines', name='Nuclear'),
                row=3, col=1
            )
            frames = [
                go.Frame(
                    data=[
                        go.Scatter(x=times[:i+1], y=exp_vals_system[:i+1], mode='lines', name='System'),
                        go.Scatter(x=times[:i+1], y=exp_vals_electron[:i+1], mode='lines', name='Electron'),
                        go.Scatter(x=times[:i+1], y=exp_vals_nuclear_avg[:i+1], mode='lines', name='Nuclear')
                    ],
                    name=f"frame{i}"
                )
                for i in range(len(times))
            ]
        else:
            exp_vals = result.expect[0]
            fig = go.Figure(
                data=[go.Scatter(x=[times[0]], y=[exp_vals[0]], mode='lines', name=observable_name)]
            )
            frames = [
                go.Frame(
                    data=[go.Scatter(x=times[:i+1], y=exp_vals[:i+1], mode='lines', name=observable_name)],
                    name=f"frame{i}"
                )
                for i in range(len(times))
            ]

        fig.frames = frames
        fig.update_layout(
            height=900 if observable_name == "multi_view" else 600,
            title="Time Evolution of Observable",
            template="plotly_white",
            showlegend=True,
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    },
                    {
                        "args": [["frame0"], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }],
                        "label": "Reset",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Time: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 50},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f"frame{i}"],
                            {"frame": {"duration": 50, "redraw": True},
                             "mode": "immediate",
                             "transition": {"duration": 50}}
                        ],
                        "label": f"{t:.1f}",
                        "method": "animate"
                    } for i, t in enumerate(times)
                ]
            }]
        )
        if observable_name == "multi_view":
            fig.update_xaxes(title_text="Time (arbitrary units)", row=3, col=1)
            fig.update_yaxes(title_text="⟨System⟩", row=1, col=1)
            fig.update_yaxes(title_text="⟨Electron⟩", row=2, col=1)
            fig.update_yaxes(title_text="⟨Nuclear⟩", row=3, col=1)
        else:
            fig.update_xaxes(title_text="Time (arbitrary units)")
            fig.update_yaxes(title_text=f"⟨{observable_name}⟩")
        st.plotly_chart(fig)
        with st.expander("Plot Context"):
            plot_explanation = st.text_area("Plot Context", 
                         f"This shows the time evolution of the observable ⟨{observable_name}⟩ over time.\n\n"
                         f"Key simulation parameters:\n"
                         f"- t_max: {t_max}\n"
                         f"- Number of steps: {num_steps}\n"
                         f"- RF Frequency: {args.get('nu_rf', 'N/A')}\n"
                         f"- RF Amplitude: {args.get('amp_rf', 'N/A')}\n"
                         f"- Initial State Index: {state_index}\n\n"
                         f"Expectation values at 10 time points:\n"
                         + "\n".join([
                             f"t = {times[i]:.2f}:"
                             + f"\nElectron: ⟨Sz⟩ = {expect(electron_ops['z'], result.states[i]):.3f}"
                             + "".join([f"\nNucleus {n+1}: ⟨Sz⟩ = {expect(nuclear_ops[n]['z'], result.states[i]):.3f}"
                                      for n in range(num_nuclei)])
                             + "\n"
                             for i in range(0, len(times), len(times)//10)[:10]
                         ]),
                         height=300,
                         key="plot_explanation")
            if st.button("Copy to Clipboard", key="copy_to_clipboard_button_2"):
                pyperclip.copy(plot_explanation)
                st.success("Plot explanation copied to clipboard!")
    else:
        st.write("No observable provided; skipping plot.")

    # Optionally generate Bloch sphere plots for selected times
    if show_bloch_spheres and times_for_bloch:
        sx_12 = sigmax()/2
        sy_12 = sigmay()/2
        sz_12 = sigmaz()/2
        cols = st.columns(len(times_for_bloch))
        for col, t_req in zip(cols, times_for_bloch):
            idx = np.argmin(np.abs(times - t_req))
            actual_t = times[idx]
            psi_t = result.states[idx]
            with col:
                st.write(f"### Bloch Spheres at t={actual_t:.3f}")
                for spin_idx in range(total_spins):
                    rho_spin = psi_t.ptrace(spin_idx)
                    ex = expect(sx_12, rho_spin)
                    ey = expect(sy_12, rho_spin)
                    ez = expect(sz_12, rho_spin)
                    if spin_idx == electron_idx:
                        label = "Electron"
                    else:
                        label = f"Nuclear {spin_idx+1}"
                    st.write(f"**{label}:** ⟨Sx⟩={ex:.3f}, ⟨Sy⟩={ey:.3f}, ⟨Sz⟩={ez:.3f}")
                    fig = plt.figure()
                    b = Bloch(fig=fig)
                    b.add_states(rho_spin)
                    b.render()
                    b.fig.suptitle(f"{label} at t={actual_t:.2f}")
                    st.pyplot(fig)
                    plt.close(fig)
    return result

# %% ChatGPT Integration Functions (if needed)

def get_simulation_context(num_nuclei, hbar, Bz, J_dd, nu_rf, amp_rf, gamma_n, q, eta, A_hf, eigenvalues):
    context = (
        f"Simulation Parameters:\n"
        f"Number of Nuclei: {num_nuclei}\n"
        f"ħ: {hbar}\n"
        f"Bz: {Bz}\n"
        f"J_dd: {J_dd}\n"
        f"RF Frequency: {nu_rf}\n"
        f"RF Amplitude: {amp_rf}\n"
        f"γ_n: {gamma_n}\n"
        f"Quadrupole coupling (q): {q}\n"
        f"Quadrupole asymmetry (η): {eta}\n"
        f"Hyperfine coupling (A_hf): {A_hf}\n\n"
        f"First few eigenvalues of the static Hamiltonian: {np.round(eigenvalues[:5],2).tolist()}\n"
    )
    return context

def ask_chatgpt(question, context):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a quantum physics and simulation expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error communicating with ChatGPT: {e}"

def plot_energy_levels_with_parameters(H_static, Bz, A_hf, J_dd, gamma_e, gamma_n, nu_rf):
    _, H_static_updated, _, _ = create_total_hamiltonian_with_rf(
        num_nuclei, Bz=Bz, gamma_e=gamma_e, J_dd=J_dd, hbar=hbar, 
        nu_rf=nu_rf, amp_rf=amp_rf, gamma_n=gamma_n, q=q, eta=eta, A_hf=A_hf
    )
    eigenvals = H_static_updated.eigenenergies()
    eigenvals_sorted = np.sort(eigenvals)
    eigenvals_sorted_rounded = np.round(eigenvals_sorted, 2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(eigenvals_sorted_rounded))),
        y=eigenvals_sorted_rounded,
        mode='lines+markers+text',
        text=[f'E={val:.2f} MHz' for val in eigenvals_sorted_rounded],
        textposition='top center',
        hovertemplate=(
            'State: %{x}<br>'
            'Energy: %{y:.2f} MHz<br>'
            f'Bz={Bz:.2f} T, A_hf={A_hf:.2f} MHz, J_dd={J_dd:.2f} MHz<br>'
            f'γ_e={gamma_e:.2f} MHz/T, γ_n={gamma_n:.2f} MHz/T, ν_rf={nu_rf:.2f} MHz'
        ),
        marker=dict(color='blue')
    ))
    fig.update_layout(
        title=f'Energy Levels (Bz={Bz:.2f} T, A_hf={A_hf:.2f} MHz, J_dd={J_dd:.2f} MHz, '
              f'γ_e={gamma_e:.2f} MHz/T, γ_n={gamma_n:.2f} MHz/T, ν_rf={nu_rf:.2f} MHz)',
        xaxis_title='State Index',
        yaxis_title='Energy (MHz)',
        template='plotly_white'
    )
    return fig
# %% Streamlit UI

st.title("Quantum Dot System Analysis: n Nuclear Spins + 1 Electron Spin")

st.markdown("""
This interactive app simulates a quantum dot system.
The Hamiltonian includes nuclear and electron Zeeman effects, nuclear quadrupole and dipole–dipole interactions, 
hyperfine coupling, and a time-dependent RF drive on the nuclei.
Adjust the parameters below to explore the system's behavior.
""")


# Create a collapsible section for the floating window
with st.expander("Hamiltonian of the Quantum Dot System", expanded=False):
    st.markdown("The Hamiltonian includes three nuclear spins (I₁, I₂, I₃) and one electron spin (S):")
    st.latex(r"""
    H = \sum_{i=1}^3 \left(H_{Z,N}^{(i)} + H_{Q,N}^{(i)}\right) + \sum_{i<j} H_{DD,N}^{(ij)} + H_{Z,e} + \sum_{i=1}^3 H_{hf}^{(i)}
    """)
    st.markdown("where:")
    st.latex(r"""
    \begin{aligned}
    H_{Z,N}^{(i)} & \quad \text{Zeeman term for nuclear spin } i \\
    H_{Q,N}^{(i)} & \quad \text{Quadrupole term for nuclear spin } i \\
    H_{DD,N}^{(ij)} & \quad \text{Dipole-dipole interaction between nuclei } i,j \\
    H_{Z,e} & \quad \text{Electron Zeeman term} \\
    H_{hf}^{(i)} & \quad \text{Hyperfine interaction between nucleus } i \text{ and electron}
    \end{aligned}
    """)

# Sidebar for Hamiltonian parameters
st.sidebar.subheader("Global Parameters")
st.session_state.num_nuclei = st.sidebar.number_input(
    "Number of Nuclei (dimensionless):", 
    0, 
    20, 
    st.session_state.get('num_nuclei', 1), 
    1
)
hilbert_space_dim = 2 * (2 ** st.session_state.num_nuclei)
st.session_state.hbar = st.sidebar.number_input(
    "ħ planck constant (dimensionless):", 
    0.0, 
    2.0, 
    st.session_state.get('hbar', 1.0), 
    0.1
)
st.session_state.Bz = st.sidebar.number_input(
    "Bz magnetic field (Tesla):", 
    0.0, 
    2.0, 
    st.session_state.get('Bz', 0.5), 
    0.1
)
st.session_state.J_dd = st.sidebar.number_input(
    "J_dd dipole-dipole coupling (MHz):", 
    0.0, 
    2.0, 
    st.session_state.get('J_dd', 0.1), 
    0.05
)
st.session_state.nu_rf = st.sidebar.number_input(
    "RF frequency (MHz):", 
    0.0, 
    50.0, 
    st.session_state.get('nu_rf', 10.0), 
    0.1
)
st.session_state.amp_rf = st.sidebar.number_input(
    "RF amplitude (dimensionless):", 
    min_value=0.0, 
    max_value=100.0, 
    value=st.session_state.get('amp_rf', 1.0),
    step=0.05
)

st.sidebar.subheader("Nuclear Parameters")
st.session_state.gamma_n = st.session_state.get('gamma_n', 1.0)
st.session_state.q = st.session_state.get('q', 0.5)
st.session_state.eta = st.session_state.get('eta', 0.2)
st.session_state.A_hf = st.session_state.get('A_hf', 0.5)

st.sidebar.markdown(
    "<span style='color:#4CAF50;'>γ_n gyromagnetic ratio (MHz/T):</span>",
    unsafe_allow_html=True
)
st.session_state.gamma_n = st.sidebar.number_input(
    "γ_n (MHz/T) input", 
    0.0, 100.0, st.session_state.gamma_n, 0.1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed"
)

st.sidebar.markdown("<span style='color:#4CAF50;'>q quadrupole coupling (MHz):</span>", unsafe_allow_html=True)
st.session_state.q = st.sidebar.number_input("q:", -2.0, 2.0, st.session_state.q, 0.1, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed")

st.sidebar.markdown(
    "<span style='color:#4CAF50;'>η quadrupole asymmetry (dimensionless):</span>",
    unsafe_allow_html=True
)
st.session_state.eta = st.sidebar.number_input(
    "η", 
    -1.0, 1.0, st.session_state.eta, 0.05, format=None, key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="collapsed"
)

st.sidebar.markdown("<span style='color:#4CAF50;'>A_hf hyperfine coupling (MHz):</span>", unsafe_allow_html=True)
st.session_state.A_hf = st.sidebar.number_input("A_hf", 0.0, 100.0, st.session_state.A_hf, 0.1, label_visibility="collapsed")

st.sidebar.subheader("Electron Parameters")
st.session_state.gamma_e = st.session_state.get('gamma_e', 28.0)
st.session_state.gamma_e = st.sidebar.number_input("γ_e gyromagnetic ratio (MHz/T):", 0.0, 50.0, st.session_state.gamma_e, 0.1)

# New: Decoherence options
st.sidebar.subheader("Decoherence Options")
include_decoherence = st.sidebar.checkbox("Include Decoherence", value=False)
# Decoherence rates for electron (in same units as simulation, e.g. MHz)
gamma_relax_e = st.sidebar.number_input("Electron Relaxation Rate (MHz)", 0.0, 10.0, 0.1, 0.1)
gamma_dephase_e = st.sidebar.number_input("Electron Dephasing Rate (MHz)", 0.0, 10.0, 0.1, 0.1)

plot_bloch_spheres = st.sidebar.checkbox("Plot Bloch Spheres for Eigenstates", key="energy_levels_bloch_spheres_checkbox")
# Define variables for use in other functions
num_nuclei = st.session_state.num_nuclei
hbar = st.session_state.hbar
Bz = st.session_state.Bz
J_dd = st.session_state.J_dd
nu_rf = st.session_state.nu_rf
amp_rf = st.session_state.amp_rf
gamma_n = st.session_state.gamma_n
q = st.session_state.q
eta = st.session_state.eta
A_hf = st.session_state.A_hf
gamma_e = st.session_state.gamma_e

with st.sidebar:
    st.sidebar.write("### Time Evolution Settings")
    state_index = st.number_input(
        "Initial State Index (dimensionless)",
        min_value=0.0,
        max_value=float(hilbert_space_dim - 1),
        value=0.0,
        step=0.5,
        help=f"Select the initial state index (0.0 to {float(hilbert_space_dim-1)})"
    )
    observable_name = st.selectbox(
        "Observable Name:", 
        ["multi_view", "electron_only"] + ["system"] + [f"nuclear_{i+1}" for i in range(num_nuclei)], 
        index=0,
        key="observable_custom"
    )
    t_max = st.number_input("Total Time (arbitrary units):", 1.0, 1000.0, 10.0, 1.0, key="t_max_custom")
    num_steps = st.number_input("Number of Time Steps (dimensionless):", 10, 1000, 200, 10, key="num_steps_custom")
    
    times_for_bloch = st.text_input("Times for Bloch Spheres (comma-separated):", "0, 5, 10", key="times_bloch_custom")
    times_for_bloch = [float(t) for t in times_for_bloch.split(",")]
    
    nu_rf_custom = st.number_input("RF Frequency (MHz):", -100.0, 100.0, 2.0, 0.1, key="nu_rf_custom")
    include_decoherence = st.checkbox("Include Decoherence", value=False, key="decoherence_checkbox")
    # Also show the checkbox in tab3
    show_bloch_spheres = st.checkbox("Show Bloch Spheres", value=False, key="time_evolution_bloch_spheres_tab")
    # Sync the two checkboxes
    # show_bloch_spheres = show_bloch_spheres or show_bloch_spheres_tab

    if include_decoherence:
        with st.sidebar:
            gamma_relax_e = st.number_input("Electron Relaxation Rate (MHz)", 0.0, 10.0, 0.1, 0.1, key="gamma_relax")
            gamma_dephase_e = st.number_input("Electron Dephasing Rate (MHz)", 0.0, 10.0, 0.1, 0.1, key="gamma_dephase")




# Initialize Hamiltonian variables in session state if they don't exist
if 'hamiltonians' not in st.session_state:
    st.session_state.hamiltonians = {
        'H_td': None,
        'H_static': None,
        'H_e_zeeman': None,
        'H_nuc_zeeman': None
    }
(st.session_state.hamiltonians['H_td'], 
 st.session_state.hamiltonians['H_static'],
 st.session_state.hamiltonians['H_e_zeeman'],
 st.session_state.hamiltonians['H_nuc_zeeman']) = create_total_hamiltonian_with_rf(
    num_nuclei=num_nuclei, Bz=Bz, gamma_e=gamma_e, J_dd=J_dd, hbar=hbar,
    nu_rf=nu_rf, amp_rf=amp_rf, gamma_n=gamma_n, q=q, eta=eta, A_hf=A_hf
)

if st.session_state.hamiltonians['H_static'] is None:
    st.warning("Hamiltonian initialization failed.")
    st.stop()

H_td = st.session_state.hamiltonians['H_td']
H_static = st.session_state.hamiltonians['H_static']
H_e_zeeman = st.session_state.hamiltonians['H_e_zeeman']
H_nuc_zeeman = st.session_state.hamiltonians['H_nuc_zeeman']

# %% Eigenstate Analysis and Energy Levels Plotting
eigenvalues, eigenstates = H_static.eigenstates()
eigenvalues = np.round(eigenvalues, 2)

eigenvals = H_static.eigenenergies()
eigenvals_sorted = np.sort(eigenvals)
ground_state_energy = eigenvals_sorted[0]
energy_diffs = eigenvals_sorted - ground_state_energy
incremental_changes = np.diff(eigenvals_sorted, prepend=eigenvals_sorted[0])
eigenvals_sorted_rounded = np.round(eigenvals_sorted, 2)
incremental_changes_rounded = np.round(incremental_changes, 2)
energy_diffs_rounded = np.round(energy_diffs, 2)

def plot_energy_levels(eigenvals_sorted_rounded, incremental_changes_rounded, energy_diffs_rounded):
    fig = go.Figure()
    num_data_points = len(eigenvals_sorted_rounded)
    show_text = num_data_points <= 32
    fig.add_trace(go.Scatter(
        x=list(range(num_data_points)),
        y=eigenvals_sorted_rounded,
        mode='lines+markers' + ('+text' if show_text else ''),
        text=eigenvals_sorted_rounded if show_text else None,
        textposition='top center',
        hovertemplate=(
            'State: %{x}<br>Energy: %{y} MHz<br>'
            'ΔE: %{customdata[0]} MHz<br>ΔE from GS: %{customdata[1]} MHz'
        ),
        customdata=np.stack((incremental_changes_rounded, energy_diffs_rounded), axis=-1)
    ))
    fig.update_layout(
        title='Sorted Energy Levels (n nuclei + 1 electron)',
        xaxis_title='State Index',
        yaxis_title='Energy (MHz)',
        template='plotly_white'
    )
    return fig

# Replace the checkbox controls with tabs

# Calculate expectation values for each eigenstate
sx = sigmax()/2
sy = sigmay()/2
sz = sigmaz()/2

# Embed operators for all spins (nuclei and electron)
electron_idx = num_nuclei  # electron is the last qubit

# Create embedded operators for each nucleus and the electron
nuclear_ops = []
for i in range(num_nuclei):
    nuclear_ops.append({
        'x': embed_operator(sx, i, num_nuclei + 1),
        'y': embed_operator(sy, i, num_nuclei + 1),
        'z': embed_operator(sz, i, num_nuclei + 1)
    })

electron_ops = {
    'x': embed_operator(sx, electron_idx, num_nuclei + 1),
    'y': embed_operator(sy, electron_idx, num_nuclei + 1),
    'z': embed_operator(sz, electron_idx, num_nuclei + 1)
}
selected_tab = st.selectbox(
    "Select View:",
    ["Energy Levels", "Interactive Energy Levels", "Time Evolution"],
    index=0,
    format_func=lambda x: f"🔍 {x}" if x == "Energy Levels" else f"📊 {x}" if x == "Interactive Energy Levels" else f"⏳ {x}"
)

if selected_tab == "Energy Levels":
    st.subheader("Energy Levels")
    fig_energy = plot_energy_levels(eigenvals_sorted_rounded, incremental_changes_rounded, energy_diffs_rounded)
    st.plotly_chart(fig_energy, use_container_width=True)

    with st.expander("Plot and Calculation Context"):
       
        
        # Calculate expectation values for each state
        state_expectations = []
        for idx, state in enumerate(eigenstates):
            # Calculate nuclear expectations
            nuclear_expects = []
            for i in range(num_nuclei):
                ex = expect(nuclear_ops[i]['x'], state)
                ey = expect(nuclear_ops[i]['y'], state)
                ez = expect(nuclear_ops[i]['z'], state)
                nuclear_expects.append((ex, ey, ez))
            
            # Calculate electron expectations
            ex_e = expect(electron_ops['x'], state)
            ey_e = expect(electron_ops['y'], state)
            ez_e = expect(electron_ops['z'], state)
            
            state_expectations.append({
                'nuclear': nuclear_expects,
                'electron': (ex_e, ey_e, ez_e)
            })
        
        context_text = f"""
        Plot Context: Sorted Energy Levels ({num_nuclei} nuclei + 1 electron)
        
        State Indices: 
        {', '.join(map(str, range(len(eigenvals_sorted_rounded))))}
        
        Energy Levels (MHz): 
        {', '.join(map(str, eigenvals_sorted_rounded))}
        
        Incremental Changes (MHz): 
        {', '.join(map(str, incremental_changes_rounded))}
        
        Energy Differences from Ground State (MHz): 
        {', '.join(map(str, energy_diffs_rounded))}
        
        Detailed State Information:
        {'\n'.join(
            f"State {idx}: Energy = {energy:.3f} MHz\n"
            f"  Electron: ⟨Sx,Sy,Sz⟩ = ({exp['electron'][0]:.3f}, {exp['electron'][1]:.3f}, {exp['electron'][2]:.3f})\n"
            + '\n'.join(
                f"  Nucleus {n+1}: ⟨Sx,Sy,Sz⟩ = ({exp['nuclear'][n][0]:.3f}, {exp['nuclear'][n][1]:.3f}, {exp['nuclear'][n][2]:.3f})"
                for n in range(num_nuclei)
            )
            for idx, (energy, exp) in enumerate(zip(eigenvals_sorted_rounded, state_expectations))
        )}
        
        Hamiltonian Input Data:
        - Number of Nuclei: {num_nuclei}
        - Magnetic Field (Bz): {Bz} T
        - Electron Gyromagnetic Ratio (gamma_e): {gamma_e} MHz/T
        - Dipole-Dipole Coupling (J_dd): {J_dd} MHz
        - Reduced Planck Constant (hbar): {hbar}
        - RF Frequency (nu_rf): {nu_rf} MHz
        - RF Amplitude (amp_rf): {amp_rf}
        - Nuclear Gyromagnetic Ratio (gamma_n): {gamma_n} MHz/T
        - Quadrupole Coupling (q): {q}
        - Quadrupole Asymmetry (eta): {eta}
        - Hyperfine Coupling (A_hf): {A_hf} MHz
        """
        
        user_prompt = st.text_area(
            "Context",
            value=context_text,
            height=300,
            key="user_prompt_text_area"
        )
        if st.button("Copy to Clipboard", key="copy_to_clipboard_button_1"):
            pyperclip.copy(user_prompt)
            st.success("User prompt copied to clipboard!")
        
    
    if plot_bloch_spheres:
        with st.container():
            st.subheader("Bloch Sphere Representations")
            all_nuclear_dms, all_electron_dms, all_energies, all_psis = [], [], [], []
            for idx in range(len(eigenvalues)):
                nuc_dms, elec_dm, energy, psi = extract_subsystems_from_eigenstate(H_static, idx)
                all_nuclear_dms.append(nuc_dms)
                all_electron_dms.append(elec_dm)
                all_energies.append(energy)
                all_psis.append(psi)
          
            sx = sigmax()/2
            sy = sigmay()/2
            sz = sigmaz()/2
            for state_idx, energy in enumerate(all_energies):
                st.markdown(f"### Eigenstate {state_idx} (Energy: {energy:.3f})")
                with st.container():
                    cols = st.columns(len(all_nuclear_dms[state_idx]) + 1)
                    for i, dm in enumerate(all_nuclear_dms[state_idx]):
                        with cols[i]:
                            st.pyplot(create_spin_bloch(dm, title=f"Nucleus {i+1}"))
                            ex, ey, ez = expect(sx, dm), expect(sy, dm), expect(sz, dm)
                            st.write(f"⟨I_x, I_y, I_z⟩ = ({ex:.2f}, {ey:.2f}, {ez:.2f})")
                    with cols[-1]:
                        st.pyplot(create_spin_bloch(all_electron_dms[state_idx], title="Electron"))
                        ex, ey, ez = expect(sx, all_electron_dms[state_idx]), expect(sy, all_electron_dms[state_idx]), expect(sz, all_electron_dms[state_idx])
                        st.write(f"⟨S_x, S_y, S_z⟩ = ({ex:.2f}, {ey:.2f}, {ez:.2f})")
                st.write("---")





if selected_tab == "Interactive Energy Levels":
    st.subheader("Interactive Energy Levels")
    DEFAULT_VALUES = {
        'Bz': 1.0,
        'A_hf': 0.5,
        'J_dd': 0.1,
        'gamma_e': 28.0,
        'gamma_n': 1.0,
        'nu_rf': 1.0
    }

    if st.button("Reset to Default Values"):
        st.session_state.reset_sliders = True
    else:
        st.session_state.reset_sliders = False

    col1, col2, col3 = st.columns(3)
    with col1:
        Bz_value = st.slider(
            "Magnetic Field (Tesla)", 
            min_value=0.0, 
            max_value=10.0, 
            value=DEFAULT_VALUES['Bz'] if st.session_state.get('reset_sliders', False) else st.session_state.get('Bz_value', DEFAULT_VALUES['Bz']),
            step=2.0,
            key='Bz_value'
        )
    with col2:
        A_hf_value = st.slider(
            "Hyperfine Coupling (MHz)", 
            min_value=0.0, 
            max_value=2.0, 
            value=DEFAULT_VALUES['A_hf'] if st.session_state.get('reset_sliders', False) else st.session_state.get('A_hf_value', DEFAULT_VALUES['A_hf']),
            step=0.5,
            key='A_hf_value'
        )
    with col3:
        J_dd_value = st.slider(
            "Dipole-Dipole Coupling (MHz)", 
            min_value=0.0, 
            max_value=2.0, 
            value=DEFAULT_VALUES['J_dd'] if st.session_state.get('reset_sliders', False) else st.session_state.get('J_dd_value', DEFAULT_VALUES['J_dd']),
            step=0.5,
            key='J_dd_value'
        )

    col4, col5, col6 = st.columns(3)
    with col4:
        gamma_e_value = st.slider(
            "Electron Gyromagnetic Ratio (MHz/T)", 
            min_value=0.0, 
            max_value=50.0, 
            value=DEFAULT_VALUES['gamma_e'] if st.session_state.get('reset_sliders', False) else st.session_state.get('gamma_e_value', DEFAULT_VALUES['gamma_e']),
            step=10.0,
            key='gamma_e_value'
        )
    with col5:
        gamma_n_value = st.slider(
            "Nuclear Gyromagnetic Ratio (MHz/T)", 
            min_value=0.0, 
            max_value=2.0, 
            value=DEFAULT_VALUES['gamma_n'] if st.session_state.get('reset_sliders', False) else st.session_state.get('gamma_n_value', DEFAULT_VALUES['gamma_n']),
            step=0.5,
            key='gamma_n_value'
        )
    with col6:
        nu_rf_value = st.slider(
            "RF Frequency (MHz)", 
            min_value=0.0, 
            max_value=2.0, 
            value=DEFAULT_VALUES['nu_rf'] if st.session_state.get('reset_sliders', False) else st.session_state.get('nu_rf_value', DEFAULT_VALUES['nu_rf']),
            step=0.5,
            key='nu_rf_value'
        )

    fig_interactive = plot_energy_levels_with_parameters(
        H_static, 
        Bz_value, 
        A_hf_value, 
        J_dd_value,
        gamma_e_value,
        gamma_n_value,
        nu_rf_value
    )
    st.plotly_chart(fig_interactive, use_container_width=True)

    # Ensure the initial values for sliders are set correctly

    # Energy Levels Animation section
    st.subheader("Energy Levels Animation")
    st.write("### Set Initial Parameters for Animation")
    col1, col2 = st.columns(2)
    with col1:
        param_to_animate = st.selectbox(
            "Select parameter to animate:",
            ["Magnetic Field (Bz)", "Hyperfine Coupling (A_hf)", "Dipole-Dipole Coupling (J_dd)",
            "Electron Gyromagnetic Ratio (γ_e)", "Nuclear Gyromagnetic Ratio (γ_n)"]
        )
        if param_to_animate == "Magnetic Field (Bz)":
            default_start, default_end = 0.0, 10.0
        elif param_to_animate == "Electron Gyromagnetic Ratio (γ_e)":
            default_start, default_end = 0.0, 50.0
        else:
            default_start, default_end = 0.0, 2.0
        start_val = st.number_input("Start value", value=default_start, step=0.1)
        end_val = st.number_input("End value", value=default_end, step=0.1)
        num_steps = st.number_input("Number of steps", value=20, min_value=5, max_value=50, step=5)
    with col2:
        st.write("Set initial values for other parameters:")
        init_Bz = st.slider("Initial Bz [T]", 0.0, 10.0, Bz_value, 2.0, disabled=param_to_animate=="Magnetic Field (Bz)")
        init_A_hf = st.slider("Initial A_hf [MHz]", 0.0, 2.0, A_hf_value, 0.5, disabled=param_to_animate=="Hyperfine Coupling (A_hf)")
        init_J_dd = st.slider("Initial J_dd [MHz]", 0.0, 2.0, J_dd_value, 0.5, disabled=param_to_animate=="Dipole-Dipole Coupling (J_dd)")
        init_gamma_e = st.slider("Initial γ_e [MHz/T]", 0.0, 50.0, gamma_e_value, 10.0, disabled=param_to_animate=="Electron Gyromagnetic Ratio (γ_e)")
        init_gamma_n = st.slider("Initial γ_n [MHz/T]", 0.0, 2.0, gamma_n_value, 0.5, disabled=param_to_animate=="Nuclear Gyromagnetic Ratio (γ_n)")

    def generate_animation_frames(param_name, start, end, steps, init_params):
        frames = []
        param_values = np.linspace(start, end, steps)
        min_energy = float('inf')
        max_energy = float('-inf')
        
        # First pass to determine global energy range
        for val in param_values:
            if param_name == "Magnetic Field (Bz)":
                current_Bz, current_A_hf, current_J_dd = val, init_params['A_hf'], init_params['J_dd']
                current_gamma_e, current_gamma_n = init_params['gamma_e'], init_params['gamma_n']
            elif param_name == "Hyperfine Coupling (A_hf)":
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], val, init_params['J_dd']
                current_gamma_e, current_gamma_n = init_params['gamma_e'], init_params['gamma_n']
            elif param_name == "Dipole-Dipole Coupling (J_dd)":
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], init_params['A_hf'], val
                current_gamma_e, current_gamma_n = init_params['gamma_e'], init_params['gamma_n']
            elif param_name == "Electron Gyromagnetic Ratio (γ_e)":
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], init_params['A_hf'], init_params['J_dd']
                current_gamma_e, current_gamma_n = val, init_params['gamma_n']
            else:  # Nuclear Gyromagnetic Ratio (γ_n)
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], init_params['A_hf'], init_params['J_dd']
                current_gamma_e, current_gamma_n = init_params['gamma_e'], val
            
            _, H_static_updated, _, _ = create_total_hamiltonian_with_rf(
                num_nuclei, 
                Bz=current_Bz, 
                gamma_e=current_gamma_e, 
                J_dd=current_J_dd, 
                hbar=hbar, 
                nu_rf=nu_rf_value, 
                amp_rf=amp_rf, 
                gamma_n=init_params['gamma_n'], 
                q=q, 
                eta=eta, 
                A_hf=current_A_hf
            )
            eigenvals = H_static_updated.eigenenergies()
            min_energy = min(min_energy, np.min(eigenvals))
            max_energy = max(max_energy, np.max(eigenvals))
            
        energy_range = max_energy - min_energy
        min_energy -= 0.1 * energy_range
        max_energy += 0.1 * energy_range

        for val in param_values:
            if param_name == "Magnetic Field (Bz)":
                current_Bz, current_A_hf, current_J_dd = val, init_params['A_hf'], init_params['J_dd']
                current_gamma_e, current_gamma_n = init_params['gamma_e'], init_params['gamma_n']
            elif param_name == "Hyperfine Coupling (A_hf)":
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], val, init_params['J_dd']
                current_gamma_e, current_gamma_n = init_params['gamma_e'], init_params['gamma_n']
            elif param_name == "Dipole-Dipole Coupling (J_dd)":
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], init_params['A_hf'], val
                current_gamma_e, current_gamma_n = init_params['gamma_e'], init_params['gamma_n']
            elif param_name == "Electron Gyromagnetic Ratio (γ_e)":
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], init_params['A_hf'], init_params['J_dd']
                current_gamma_e, current_gamma_n = val, init_params['gamma_n']
            else:  # Nuclear Gyromagnetic Ratio (γ_n)
                current_Bz, current_A_hf, current_J_dd = init_params['Bz'], init_params['A_hf'], init_params['J_dd']
                current_gamma_e, current_gamma_n = init_params['gamma_e'], val
            
            _, H_static_updated, _, _ = create_total_hamiltonian_with_rf(
                num_nuclei, 
                Bz=current_Bz, 
                gamma_e=current_gamma_e, 
                J_dd=current_J_dd, 
                hbar=hbar, 
                nu_rf=nu_rf_value, 
                amp_rf=amp_rf, 
                gamma_n=init_params['gamma_n'], 
                q=q, 
                eta=eta, 
                A_hf=current_A_hf
            )
            eigenvals = H_static_updated.eigenenergies()
            eigenvals_sorted = np.sort(eigenvals)
            eigenvals_sorted_rounded = np.round(eigenvals_sorted, 2)
            
            param_label = {
                "Magnetic Field (Bz)": f"Bz = {val:.2f} T",
                "Hyperfine Coupling (A_hf)": f"A_hf = {val:.2f} MHz",
                "Dipole-Dipole Coupling (J_dd)": f"J_dd = {val:.2f} MHz",
                "Electron Gyromagnetic Ratio (γ_e)": f"γ_e = {val:.2f} MHz/T",
                "Nuclear Gyromagnetic Ratio (γ_n)": f"γ_n = {val:.2f} MHz/T"
            }[param_name]
            
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=list(range(len(eigenvals_sorted_rounded))),
                    y=eigenvals_sorted_rounded,
                    mode='lines+markers+text',
                    text=[f'{v:.2f}' for v in eigenvals_sorted_rounded],
                    textposition='top center',
                    name=param_label,
                    hovertemplate=(
                        'State: %{x}<br>'
                        'Energy: %{y:.2f} MHz<br>'
                        f'Bz = {current_Bz:.2f} T<br>'
                        f'A_hf = {current_A_hf:.2f} MHz<br>'
                        f'J_dd = {current_J_dd:.2f} MHz<br>'
                        f'γ_e = {current_gamma_e:.2f} MHz/T<br>'
                        f'γ_n = {current_gamma_n:.2f} MHz/T'
                    ),
                    textfont=dict(size=10)
                )],
                name=str(val)
            ))
        return frames, param_values, min_energy, max_energy

    if st.button("Generate Animation"):
        with st.spinner("Generating animation..."):
            init_params = {
                'Bz': init_Bz,
                'A_hf': init_A_hf,
                'J_dd': init_J_dd,
                'gamma_e': init_gamma_e,
                'gamma_n': init_gamma_n
            }
            frames, param_values, min_energy, max_energy = generate_animation_frames(
                param_to_animate, start_val, end_val, num_steps, init_params
            )
            fig = go.Figure(
                data=[frames[0].data[0]],
                layout=go.Layout(
                    title=f"Energy Levels vs {param_to_animate}",
                    xaxis=dict(title="State Index"),
                    yaxis=dict(
                        title="Energy (MHz)",
                        range=[min_energy, max_energy],
                        showgrid=True,
                        zeroline=True,
                        showline=True,
                        showticklabels=True,
                        fixedrange=True
                    ),
                    updatemenus=[dict(
                        type="buttons",
                        showactive=False,
                        buttons=[dict(
                            label="Play",
                            method="animate",
                            args=[None, dict(
                                frame=dict(duration=1000, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                                transition=dict(duration=500)
                            )]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode="immediate"
                            )]
                        )]
                    )],
                    sliders=[dict(
                        currentvalue=dict(
                            font=dict(size=12),
                            prefix=f"{param_to_animate} = ",
                            suffix={
                                "Magnetic Field (Bz)": " T",
                                "Hyperfine Coupling (A_hf)": " MHz",
                                "Dipole-Dipole Coupling (J_dd)": " MHz",
                                "Electron Gyromagnetic Ratio (γ_e)": " MHz/T",
                                "Nuclear Gyromagnetic Ratio (γ_n)": " MHz/T"
                            }[param_to_animate],
                            visible=True,
                            xanchor="right"
                        ),
                        transition=dict(duration=500),
                        steps=[dict(
                            args=[[f"{val}"], dict(
                                frame=dict(duration=1000, redraw=False),
                                mode="immediate",
                                transition=dict(duration=500)
                            )],
                            label=f"{val:.2f}",
                            method="animate"
                        ) for val in param_values]
                    )],
                    showlegend=False
                ),
                frames=frames
            )
            fig.update_layout(
                height=600,
                hovermode='closest',
                margin=dict(t=100, b=50),
                transition=dict(duration=500, easing="linear"),
                xaxis=dict(showgrid=True, zeroline=True, showline=True, showticklabels=True),
                yaxis_range=[min_energy, max_energy]
            )
            st.plotly_chart(fig, use_container_width=True)

if selected_tab == "Time Evolution":
    # Add the checkbox to both sidebar and tab when tab3 is active
    
    st.subheader("Time Evolution")

    
    # Define collapse operators if decoherence is included
    collapse_ops = []
    if include_decoherence:
        # For electron relaxation: use electron lowering operator.
        sx = sigmax() / 2
        sy = sigmay() / 2
        sz = sigmaz() / 2
        sm = sigmam()
        electron_lower = embed_operator(sm, num_nuclei, num_nuclei+1)
        collapse_ops.append(np.sqrt(gamma_relax_e) * electron_lower)
        # For pure dephasing: use sz operator (optional)
        electron_dephase = embed_operator(sz, num_nuclei, num_nuclei+1)
        collapse_ops.append(np.sqrt(gamma_dephase_e) * electron_dephase)
    
    res = evolve_and_plot_time_dep(
        H=H_td,
        state_index=state_index,
        t_max=t_max,
        num_steps=num_steps,
        observable_name=observable_name,
        times_for_bloch=times_for_bloch,
        args={"nu_rf": nu_rf_custom, "amp_rf": amp_rf},
        include_decoherence=include_decoherence,
        collapse_ops=collapse_ops,
        show_bloch_spheres=show_bloch_spheres
    )
    
