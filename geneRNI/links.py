"""tools related to links management


"""

import operator

import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None


def format_links(
            links_matrix,
            gene_names
    ):

    """Gets the regulatory links in a matrix ({gene1-gene1, gene1-gene2, ...; gene2-gene1, gene2-gene2, etc}) converts it to a df.
    
    Parameters
    ----------
    
    gene_names: list of strings, optional
        List of length p, where p is the number of rows/columns in VIM, containing the names of the genes. The i-th item of gene_names must correspond to the i-th row/column of VIM. When the gene names are not provided, the i-th gene is named Gi.
        default: None
        
    regulators: list of strings, optional
        List containing the names of the candidate regulators. When a list of regulators is provided, the names of all the genes must be provided (in gene_names), and the returned list contains only edges directed from the candidate regulators. When regulators is set to 'all', any gene can be a candidate regulator.
        default: 'all'
        
    Returns
    -------
    
    A df with the format:            
        Regulator   Target     Weight    Sign
    """

    # reformat to i to j score. no gene to itself regulatory score
    i_j_links = [(i, j, score) for (i, j), score in np.ndenumerate(links_matrix) if i != j]

    # Write the ranked list of edges
    regs = []
    targs = []
    scores = []
    for i in range(len(i_j_links)):
        (regulator_idx, target_idx, score) = i_j_links[i]
        regs.append(gene_names[regulator_idx])
        targs.append(gene_names[target_idx])
        scores.append(score)

    df = pd.DataFrame()
    df['Regulator'] = regs
    df['Target'] = targs
    df['Weight'] = scores
    return df


def output(links: pd.DataFrame, file_name: str):
    if file_name is not None and not isinstance(file_name, str):
        raise ValueError('input argument file_name must be a string')
    links.to_csv(file_name, index=False)


def sort_links(links, sorted_gene_names, regulator_tag='Regulator', target_tag='Target', weight_tag='Weight'):
    """ Sorts links in based on gene numbers. The output looks like:
        Regulator    Target     Weight
        G1             G2         0.5
        G1             G3         0.8
    links --  Target Regulator Weight as a database
    sorted_gene_names -- gene names sorted
    """
    # TODO: how to deal with missing genes

    for i, gene in enumerate(sorted_gene_names):
        df_gene = links.loc[links[regulator_tag] == gene]
        sorted_gene_names_a = [x for x in sorted_gene_names if x != gene]
        df_gene.loc[:, target_tag] = pd.Categorical(df_gene[target_tag], sorted_gene_names_a)
        df_gene_sorted = df_gene.sort_values(target_tag)
        if i == 0:
            sorted_links = df_gene_sorted
        else:
            sorted_links = pd.concat([sorted_links, df_gene_sorted], axis=0, ignore_index=True)
    return sorted_links
