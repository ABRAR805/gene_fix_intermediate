def generate_repair_plan(gene, mutation, enzyme=None, pam=None, gRNA=None, strategy=None, ssODN=None, validation=None):
    """
    Generates a biologically consistent and detailed CRISPR repair plan
    for the specified gene and mutation.

    Args:
    gene (str): The name of the gene (e.g., 'TP53', 'KRAS').
    mutation (str): The mutation in HGVS protein format (e.g., 'p.R175H').
    enzyme (str): The CRISPR enzyme to use (e.g., 'SpCas9', 'SaCas9'). Default is None.
    pam (str): The PAM sequence (e.g., 'NGG'). Default is None.
    gRNA (str): The gRNA sequence. Default is None.
    strategy (str): The strategy for editing (e.g., 'HDR', 'Prime Editing'). Default is None.
    ssODN (str): The ssODN (single-strand oligo DNA) or pegRNA sequence. Default is None.
    validation (str): Validation steps (e.g., 'Sequence the edited locus'). Default is None.

    Returns:
    str: A detailed CRISPR repair plan.
    """

    # Predefined CRISPR design per gene
    gene_data = {
        "TP53": {
            "enzyme": "SpCas9",
            "pam": "NGG",
            "gRNA": "CAGGGCAGCTACGGTTTCCG",
            "remark": "Tumor suppressor gene – off-target effects must be minimized.",
        },
        "KRAS": {
            "enzyme": "SaCas9",
            "pam": "NNGRRT",
            "gRNA": "GTTGGAGCTGGTGGCGTAGG",
            "remark": "Focus on exon 2 targeting for G12 variants.",
        },
        "BRCA1": {
            "enzyme": "AsCas12a",
            "pam": "TTTV",
            "gRNA": "TTTGAGCTGCTTCTCAGTTC",
            "remark": "Used for correcting pathogenic variants in breast/ovarian cancers.",
        },
        "BRCA2": {
            "enzyme": "LbCas12a",
            "pam": "TTTV",
            "gRNA": "TTTCTCTGAGATAGGTCACC",
            "remark": "HDR with long homology arms preferred.",
        },
        "EGFR": {
            "enzyme": "SpCas9-HF1",
            "pam": "NGG",
            "gRNA": "AGGAGCTGTTCGGAATCCCA",
            "remark": "Target known hotspots like exon 19 or 21 in lung cancer.",
        },
    }

    # Default data if the gene is not in the predefined set
    default_data = {
        "enzyme": "SpCas9",
        "pam": "NGG",
        "gRNA": "GCTGAGTCCGTGACCTGTTG",
        "remark": "Generic CRISPR repair strategy. Gene-specific tools recommended.",
    }

    # Get gene-specific data or fallback to default
    gene_info = gene_data.get(gene.upper(), default_data)

    # If the custom enzyme, PAM, or gRNA is provided, override defaults
    enzyme = enzyme or gene_info["enzyme"]
    pam = pam or gene_info["pam"]
    gRNA = gRNA or gene_info["gRNA"]
    remark = gene_info["remark"]

    # Infer mutation type
    if "fs" in mutation.lower() or "frameshift" in mutation.lower():
        mutation_type = "Frameshift"
        strategy = strategy or "Prime Editing"
        ssODN = ssODN or f"Design pegRNA to correct frameshift caused by {mutation}"
    elif "*" in mutation or "X" in mutation.upper():
        mutation_type = "Nonsense"
        strategy = strategy or "HDR"
        ssODN = ssODN or f"ssODN to restore codon lost in {mutation} nonsense mutation"
    elif "del" in mutation.lower():
        mutation_type = "Deletion"
        strategy = strategy or "HDR or NHEJ"
        ssODN = ssODN or f"ssODN to reinsert deleted bases at {mutation}"
    elif "ins" in mutation.lower():
        mutation_type = "Insertion"
        strategy = strategy or "HDR"
        ssODN = ssODN or f"ssODN to remove inserted sequence from {mutation}"
    elif ">" in mutation or mutation.startswith("p."):
        mutation_type = "Missense"
        strategy = strategy or "HDR"
        ssODN = ssODN or f"ssODN to correct the missense mutation {mutation}"
    else:
        mutation_type = "Unknown"
        strategy = strategy or "HDR"
        ssODN = ssODN or f"ssODN designed based on sequence flanking {mutation}"

    # Validation fallback
    validation = validation or "Sequence the edited locus to confirm correction."

    # Build the repair plan with all details
    repair_plan = (
        f"🧬 CRISPR Repair Plan for Gene: {gene.upper()}\n"
        f"🔬 Mutation: {mutation} ({mutation_type})\n\n"
        f"1. **Enzyme**: {enzyme}\n"
        f"2. **PAM Sequence**: {pam}\n"
        f"3. **gRNA**: {gRNA}\n"
        f"4. **Editing Strategy**: {strategy}\n"
        f"5. **ssODN / pegRNA**: {ssODN}\n"
        f"6. **Validation**: {validation}\n"
        f"7. **Note**: {remark}\n"
        f"8. **Mutation Details**: {mutation_type} mutation requires specific care.\n"
        f"9. **Targeting Recommendations**: If applicable, ensure proper exon targeting for {gene}.\n"
    )

    return repair_plan
