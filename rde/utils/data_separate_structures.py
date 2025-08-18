import torch
from torch.utils.data import DataLoader
from rde.utils.data import PaddingCollate


class SeparateStructuresCollate:
    """
    Custom collate function for datasets that return separate wildtype and mutant structures.
    """
    
    def __init__(self):
        self.padding_collate = PaddingCollate()
    
    def __call__(self, batch):
        # Separate wildtype and mutant data
        wildtype_batch = [item['wildtype'] for item in batch]
        mutant_batch = [item['mutant'] for item in batch]
        
        # Collate wildtype and mutant separately
        wildtype_collated = self.padding_collate(wildtype_batch)
        mutant_collated = self.padding_collate(mutant_batch)
        
        # Create flattened batch for compatibility with regular model
        # Use wildtype data as the base and add mutant amino acids
        flattened_batch = wildtype_collated.copy()
        flattened_batch['aa_mut'] = mutant_collated['aa']
        
        # Return both nested and flattened structures
        return {
            'wildtype': wildtype_collated,
            'mutant': mutant_collated,
            # Flattened structure for compatibility
            **flattened_batch
        }


def create_separate_structures_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """
    Create a DataLoader for datasets with separate wildtype and mutant structures.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=SeparateStructuresCollate(),
        num_workers=num_workers
    ) 