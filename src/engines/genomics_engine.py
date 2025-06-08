# Copyright (c) 2025 TanukiMCP Orchestra
# Licensed under Non-Commercial License - Commercial use requires approval from TanukiMCP
# Contact tanukimcp@gmail.com for commercial licensing inquiries

"""
Genomics and Bioinformatics Engine

Provides production-quality computational tools for genomics, molecular biology,
and bioinformatics. This engine leverages the Biopython library for robust and
accurate biological sequence analysis. All functions are fully implemented.
"""

import logging
from typing import Dict, List, Any

# Biopython is a hard requirement for this engine
try:
    from Bio.Seq import Seq
    from Bio.SeqUtils import gc_fraction
    from Bio.Data import CodonTable
    from Bio.Align import PairwiseAligner
    from Bio import Entrez, SeqIO
except ImportError:
    raise ImportError("Biopython is not installed. Please install it with 'pip install biopython'")

logger = logging.getLogger(__name__)

# Entrez requires an email for identification to NCBI
Entrez.email = "tanukimcp@gmail.com"

class GenomicsEngine:
    """
    Implements computational tools for the genomics and bioinformatics domain.
    """
    
    def __init__(self):
        self.name = "Genomics and Bioinformatics Engine"
        self.version = "1.0.0"
        self.supported_calculations = [
            "transcribe_dna_to_rna",
            "translate_rna_to_protein",
            "calculate_gc_content",
            "fetch_genbank_record",
            "pairwise_sequence_alignment"
        ]
        # Initialize a standard pairwise aligner
        self.aligner = PairwiseAligner()
        self.aligner.mode = 'global' # Using Needleman-Wunsch algorithm

    def transcribe_dna_to_rna(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Transcribes a DNA sequence into an RNA sequence."""
        try:
            dna_sequence = Seq(parameters['dna_sequence'].upper())
            rna_sequence = dna_sequence.transcribe()
            return {"rna_sequence": str(rna_sequence)}
        except KeyError:
            return {"error": "Missing 'dna_sequence' parameter."}
        except Exception as e:
            return {"error": f"An error occurred during transcription: {e}"}

    def translate_rna_to_protein(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Translates an RNA sequence into a protein (amino acid) sequence."""
        try:
            rna_sequence = Seq(parameters['rna_sequence'].upper())
            # Ensure the sequence length is a multiple of 3
            if len(rna_sequence) % 3 != 0:
                return {"error": "RNA sequence length must be a multiple of three."}
            protein_sequence = rna_sequence.translate()
            return {"protein_sequence": str(protein_sequence)}
        except KeyError:
            return {"error": "Missing 'rna_sequence' parameter."}
        except CodonTable.TranslationError as e:
             return {"error": f"Translation error: {e}. Ensure the sequence is valid RNA and does not contain premature stop codons."}
        except Exception as e:
            return {"error": f"An error occurred during translation: {e}"}

    def calculate_gc_content(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Calculates the GC content of a DNA or RNA sequence."""
        try:
            sequence = Seq(parameters['sequence'].upper())
            gc_content = gc_fraction(sequence) * 100
            return {"gc_content_percentage": round(gc_content, 2)}
        except KeyError:
            return {"error": "Missing 'sequence' parameter."}
        except Exception as e:
            return {"error": f"An error occurred: {e}"}

    def fetch_genbank_record(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Fetches and parses a record from the NCBI GenBank database."""
        try:
            genbank_id = parameters['genbank_id']
            handle = Entrez.efetch(db="nucleotide", id=genbank_id, rettype="gb", retmode="text")
            record = SeqIO.read(handle, "genbank")
            handle.close()

            features = []
            for feature in record.features:
                if feature.type not in ["source", "misc_feature"]: # Filter out less informative features
                    features.append({
                        "type": feature.type,
                        "location": str(feature.location),
                        "qualifiers": {k: v[0] if len(v)==1 else v for k, v in feature.qualifiers.items()}
                    })

            return {
                "id": record.id,
                "name": record.name,
                "description": record.description,
                "sequence_length": len(record.seq),
                "features": features
            }
        except KeyError:
            return {"error": "Missing 'genbank_id' parameter."}
        except Exception as e:
            logger.error(f"Failed to fetch GenBank record {parameters.get('genbank_id', '')}: {e}")
            return {"error": f"Could not retrieve or parse GenBank record. It may not exist or there was a network issue."}

    def pairwise_sequence_alignment(self, parameters: Dict[str, str]) -> Dict[str, Any]:
        """Performs a pairwise global alignment of two sequences."""
        try:
            seq1 = Seq(parameters['sequence1'])
            seq2 = Seq(parameters['sequence2'])
            
            alignments = self.aligner.align(seq1, seq2)
            # Get the first and best alignment
            best_alignment = alignments[0]

            return {
                "alignment_score": best_alignment.score,
                "aligned_sequence_1": str(best_alignment[0]),
                "aligned_sequence_2": str(best_alignment[1]),
                "algorithm": "Needleman-Wunsch (Global)"
            }
        except KeyError:
            return {"error": "Missing 'sequence1' or 'sequence2' parameter."}
        except IndexError:
             return {"error": "Could not generate an alignment for the given sequences."}
        except Exception as e:
            return {"error": f"An error occurred during alignment: {e}"} 