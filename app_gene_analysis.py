#
# Application: gene analysis
#

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import requests
import sys

def get_protein_info(peptide_id_string: str) -> dict:
    """
    Fetches protein and gene names from UniProt using a peptide identifier string.

    Args:
        peptide_id_string: The full string, e.g., "STV... @ sp|O00330|ODPX".

    Returns:
        A dictionary containing the protein name and gene name, or an error message.
    """
    try:
        # Extract the identifier part after the '@' symbol
        identifier_part = peptide_id_string.split('@')[1].strip()
        # Split by '|' and get the middle part, which is the accession number
        accession_id = identifier_part.split('|')[1]
    except IndexError:
        return {"error": "Invalid input format. Expected '...@ sp|ACCESSION|...'"}

    # The modern UniProt REST API endpoint
    api_url = f"https://rest.uniprot.org/uniprotkb/{accession_id}"

    # Define the fields (data) we want to retrieve
    params = {
        "fields": "protein_name,gene_primary",
        "format": "json"
    }

    try:
        response = requests.get(api_url, params=params)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        data = response.json()

        # Extract the required information from the JSON response
        protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
        
        # Genes are in a list, so we safely get the first one
        genes = data.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "N/A") if genes else "N/A"

        return {
            "Accession ID": accession_id,
            "Protein Name": protein_name,
            "Gene Name": gene_name
        }

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {"error": f"Protein with accession ID '{accession_id}' not found."}
        else:
            return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network error occurred: {req_err}"}

def get_protein_info_with_ensg(peptide_id_string: str) -> dict:
    """
    Fetches protein, gene, and ENSG codes from UniProt using the requests library.

    Args:
        accession_id: A UniProt accession ID, e.g., "P04637".

    Returns:
        A dictionary with the requested information or an error message.
    """
    try:
        # Extract the identifier part after the '@' symbol
        identifier_part = peptide_id_string.split('@')[1].strip()
        # Split by '|' and get the middle part, which is the accession number
        accession_id = identifier_part.split('|')[1]
    except IndexError:
        return {"error": "Invalid input format. Expected '...@ sp|ACCESSION|...'"}

    api_url = f"https://rest.uniprot.org/uniprotkb/{accession_id}"

    # We still request 'xref_ensembl' as it contains the ENSG code
    params = {
        "fields": "protein_name,gene_primary,xref_ensembl",
        "format": "json"
    }

    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        
        data = response.json()

        # --- Extract basic information ---
        protein_name = data.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
        
        genes = data.get("genes", [])
        gene_name = genes[0].get("geneName", {}).get("value", "N/A") if genes else "N/A"

        # --- Extract Ensembl Gene IDs (ENSG codes) ---
        cross_refs = data.get("uniProtKBCrossReferences", [])
        ensg_codes = set() # Use a set to automatically handle duplicates

        for xref in cross_refs:
            if xref.get('database') == 'Ensembl':
                # The ENSG is in the 'properties' list
                for prop in xref.get('properties', []):
                    if prop.get('key') == 'GeneId':
                        ensg_codes.add(prop.get('value'))

        return {
            "Accession ID": accession_id,
            "Protein Name": protein_name,
            "Gene Name": gene_name,
            "ENSG Codes": list(ensg_codes) if ensg_codes else "N/A"
        }

    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            return {"error": f"Protein with accession ID '{accession_id}' not found."}
        return {"error": f"HTTP error occurred: {http_err}"}
    except requests.exceptions.RequestException as req_err:
        return {"error": f"A network error occurred: {req_err}"}
    

if __name__ == "__main__":

    data = pd.read_csv('./peptides/peptides_ATE_results.csv')
    data[(data.lower_ci_clipped > 0) & (data.upper_ci_clipped > 0)].shape[0] + data[(data.lower_ci_clipped < 0) & (data.upper_ci_clipped < 0)].shape[0]
    data[(data.lower_ci_dr > 0) & (data.upper_ci_dr > 0)].shape[0] + data[(data.lower_ci_dr < 0) & (data.upper_ci_dr < 0)].shape[0]

    dr_clip_results = np.unique(data[(data.lower_ci_clipped > 0) & (data.upper_ci_clipped > 0)].peptide.tolist() + data[(data.lower_ci_clipped < 0) & (data.upper_ci_clipped < 0)].peptide.tolist())
    dr_results = np.unique(data[(data.lower_ci_dr > 0) & (data.upper_ci_dr > 0)].peptide.tolist() + data[(data.lower_ci_dr < 0) & (data.upper_ci_dr < 0)].peptide.tolist())

    # Intersection
    np.intersect1d(dr_clip_results, dr_results).shape[0]
    # Difference
    diff_peps = np.setdiff1d(dr_clip_results, dr_results)

    genes_diff = []
    for peptide_string in diff_peps:
        info = get_protein_info_with_ensg(peptide_string)['ENSG Codes']
        # info = get_protein_info(peptide_string)['Gene Name']
        print(peptide_string, info)
        for i in info:
            genes_diff.append(i)

    np.unique(genes_diff).shape[0]
    genes_maya = pd.read_csv('./peptides/TWAS validation-Table 1.csv')

    # remove '.number' from ensembl ids
    genes_diff = [gene.split('.')[0] for gene in genes_diff]
    genes_maya.gene = genes_maya.gene.str.split('.').str[0]

    # genes in diff and in maya
    both = list(set(genes_diff) & set(genes_maya.gene.tolist()))

    # pvalue of both
    genes_maya[genes_maya.gene.isin(both)].pvalue

    genes_clip = []
    for peptide_string in dr_clip_results:
        info = get_protein_info(peptide_string)['Gene Name']
        print(peptide_string, info)
        genes_clip.append(info)

    # genes in dr_clip and in maya
    list(set(genes_clip) & set(genes_maya.alias.tolist()))

    genes_dr = []
    for peptide_string in dr_results:
        info = get_protein_info(peptide_string)['Gene Name']
        print(info)
        genes_dr.append(info)

    difference = list(set(genes_clip) - set(genes_dr))
    len(difference)
    len(genes_dr)

    # print peptides associated to genes in difference
    k=0
    for gene in difference:
        # position in genes_clip
        positions = [i for i, x in enumerate(genes_clip) if x == gene]
        for pos in positions:
            print(k, gene, dr_clip_results[pos])
            k+=1


    # all_genes = []
    # for peptide_string in data.peptide.tolist():
    #     info = get_protein_info(peptide_string)['Gene Name']
    #     all_genes.append(info)

    # gp = gprofiler.GProfiler(return_dataframe=True)
    # go = gp.profile(organism='hsapiens',
    #             query=genes_clip,
    #             domain_scope="custom",
    #             background=all_genes,
    #             significance_threshold_method='fdr',
    #             user_threshold=0.05)
    
