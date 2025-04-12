import re

def parse_mutation(mutation):
    """
    Parse the mutation string in HGVS format and extract relevant details.
    Args:
    mutation (str): Mutation in HGVS format (e.g., p.R175H, c.5382insC)
    Returns:
    dict: Parsed mutation information containing type, position, and sequence changes.
    """
    mutation_patterns = {
        "missense": r"^p\.(\w+)(\d+)(\w+)$",      # p.R175H or p.G12V
        "nonsense": r"^p\.(\w+)(\d+)(\*|\?|\$)$", # p.R175* or p.G12*
        "insertion": r"^c\.(\d+)ins([A-Za-z0-9]+)$",  # c.5382insC
        "deletion": r"^c\.(\d+)del([A-Za-z0-9]+)$"   # c.5382delA
    }

    for mut_type, pattern in mutation_patterns.items():
        match = re.match(pattern, mutation)
        if match:
            data = {'mutation_type': mut_type}
            if mut_type in ["missense", "nonsense"]:
                data.update({
                    'amino_acid': match.group(1),
                    'position': match.group(2),
                    'new_amino_acid': match.group(3)
                })
            elif mut_type == "insertion":
                data.update({
                    'position': match.group(1),
                    'inserted_sequence': match.group(2)
                })
            elif mut_type == "deletion":
                data.update({
                    'position': match.group(1),
                    'deleted_sequence': match.group(2)
                })
            return data
    return {"error": "Invalid HGVS mutation format"}


def generate_crispr_plan(gene, mutation, enzyme, pam, grna):
    """
    Generate a detailed CRISPR repair plan using the mutation parsing function.
    """
    mutation_info = parse_mutation(mutation)
    if "error" in mutation_info:
        return f"❌ Error: {mutation_info['error']}"

    plan = f"🧬 CRISPR Repair Plan for Gene: {gene}\n"
    plan += f"🔬 Mutation: {mutation} ({mutation_info['mutation_type']})\n"
    plan += f"1. Enzyme: {enzyme}\n"
    plan += f"2. PAM Sequence: {pam}\n"
    plan += f"3. gRNA: {grna}\n"

    # Strategy and repair based on mutation type
    strategy = "HDR" if mutation_info['mutation_type'] in ["missense", "insertion", "deletion"] else "Base Editing"
    plan += f"4. Repair Strategy: {strategy}\n"

    if mutation_info['mutation_type'] == "missense":
        ssodn = (f"ssODN to change amino acid {mutation_info['amino_acid']}{mutation_info['position']} "
                 f"to {mutation_info['new_amino_acid']}.")
    elif mutation_info['mutation_type'] == "nonsense":
        ssodn = (f"pegRNA to convert premature stop codon at position {mutation_info['position']} "
                 f"back to functional amino acid (base editing).")
    elif mutation_info['mutation_type'] == "insertion":
        ssodn = (f"ssODN to remove inserted sequence '{mutation_info['inserted_sequence']}' "
                 f"at position {mutation_info['position']}.")
    elif mutation_info['mutation_type'] == "deletion":
        ssodn = (f"ssODN to restore deleted sequence '{mutation_info['deleted_sequence']}' "
                 f"at position {mutation_info['position']}.")
    else:
        ssodn = "Custom repair template needed."

    plan += f"5. ssODN / pegRNA: {ssodn}\n"
    plan += f"6. Validation: Sequence the edited locus to confirm correction.\n"
    plan += f"7. Mutation Details: {mutation_info}\n"
    plan += f"8. Safety: Evaluate off-target effects using in silico tools and validate with experimental assays."

    return plan
