import numpy as np


def compute_net_force_on_geom(ncon, contact_list, efc_force, geom_id):
    """
    Computes the *net* force on any given geom. Ignores which other geoms
    are involved in the contact.

    Returns:
      - total_force_world : sum of all per-contact forces on the specified geom (world coords)
      - force_magnitude   : scalar norm of that force
      - force_direction   : unit vector in world coordinates
    """
    total_force_world = np.zeros(3)
    # Sum over all contacts that involve geom_id
    for i in range(ncon):
        contact = contact_list[i]
        if geom_id not in [contact["geom1"], contact["geom2"]]:
            continue

        dim = contact["dim"]
        efc_addr = contact["efc_address"]

        # Get local contact forces from efc_force
        contact_forces_local = efc_force[efc_addr : efc_addr + dim]

        # The 3×3 contact frame for normal + friction directions
        frame = np.array(contact["frame"]).reshape((3, 3))

        # Convert local forces to world space
        force_world = frame[:, :dim].dot(contact_forces_local)
        total_force_world += force_world

    # Compute magnitude & direction
    force_magnitude = np.linalg.norm(total_force_world)
    if force_magnitude > 1e-12:
        force_direction = total_force_world / force_magnitude
    else:
        force_direction = np.zeros(3)

    return total_force_world, force_magnitude, force_direction
