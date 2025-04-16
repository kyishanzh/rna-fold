import numpy as np
import csv
from os import path
import polars as pl
import yaml

class Config:
    """
    A configuration class that converts dictionary entries into class attributes.
    
    This class provides a convenient way to access configuration parameters
    as object attributes rather than dictionary items.
    
    Attributes:
        entries (dict): Original dictionary of configuration parameters
    """
    def __init__(self, **entries):
        """
        Initialize Config with dictionary entries.
        
        Args:
            **entries: Arbitrary keyword arguments that will become class attributes
        """
        self.__dict__.update(entries)
        self.entries = entries

    def print(self):
        """Print all configuration entries."""
        print(self.entries)

def drop_pk5090_duplicates(df):
    """
    Filter a dataframe to handle PK50 and PK90 dataset duplicates.
    
    This function specifically filters for:
    - PK50_AltChemMap_NovaSeq entries
    - PK90_Twist_epPCR entries
    And combines them with non-PK entries.
    
    Args:
        df (pl.DataFrame): Input Polars DataFrame containing RNA data
        
    Returns:
        pl.DataFrame: Filtered DataFrame containing specific PK50/PK90 entries
        
    Raises:
        AssertionError: If the filtered PK datasets don't match expected sizes
    """
    pk50_filter=df['dataset_name'].str.starts_with('PK50')
    pk90_filter=df['dataset_name'].str.starts_with('PK90')
    no_pk_df=df.filter((~pk50_filter) & (~pk90_filter))
    pk50_df=df.filter(df['dataset_name'].str.starts_with('PK50_AltChemMap_NovaSeq'))
    pk90_df=df.filter(df['dataset_name'].str.starts_with('PK90_Twist_epPCR'))

    assert len(pk50_df)==2729*2
    assert len(pk90_df)==2173*2

    new_df=pl.concat([no_pk_df,pk50_df,pk90_df])

    return new_df

def dataset_dropout(dataset_name, train_indices, dataset2drop):
    """
    Remove specific dataset examples from training indices.
    
    Args:
        dataset_name (Union[pl.Series, np.ndarray]): Array of dataset names
        train_indices (np.ndarray): Array of training indices
        dataset2drop (str): Name of dataset to exclude
        
    Returns:
        np.ndarray: Updated training indices with specified dataset removed
        
    Example:
        >>> train_indices = dataset_dropout(names, indices, "PK50")
    """
    #dataset_name=pl.Series(dataset_name)
    dataset_filter=pl.Series(dataset_name).str.starts_with(dataset2drop)
    dataset_filter=dataset_filter.to_numpy()

    dropout_indcies=set(np.where(dataset_filter==False)[0])
    # print(dropout_indcies)
    # exit()


    print(f"number of training examples before droppint out {dataset2drop}")
    print(train_indices.shape)
    before=len(train_indices)

    train_indices=set(train_indices).intersection(set(np.where(dataset_filter==False)[0]))
    train_indices=np.array(list(train_indices))

    print(f"number of training examples after droppint out {dataset2drop}")
    print(len(train_indices))
    after=len(train_indices)
    print(before-after," sequences are dropped")


    # print(set([dataset_name[i] for i in train_indices]))
    # print(len(set([dataset_name[i] for i in train_indices])))
    # exit()

    return train_indices

def get_pl_train(pl_train, seq_length=457):
    """
    Process training data into required format for model training.
    
    This function performs several operations:
    1. Removes duplicates and sorts data
    2. Creates label names for each sequence position
    3. Formats sequences, labels, and metadata
    4. Sets signal-to-noise ratio to fixed value
    
    Args:
        pl_train (pl.DataFrame): Polars DataFrame containing training data
        seq_length (int, optional): Length of RNA sequences. Defaults to 457.
        
    Returns:
        dict: Dictionary containing:
            - sequences: List of RNA sequences
            - sequence_ids: List of sequence identifiers
            - labels: Array of reactivity labels
            - errors: Array of error values
            - SN: Array of signal-to-noise ratios
    """
    print(f"before filtering pl_train has shape {pl_train.shape}")
    pl_train=pl_train.unique(subset=["sequence_id", "experiment_type"]).sort(["sequence_id", "experiment_type"])
    print(f"after filtering pl_train has shape {pl_train.shape}")
    #seq_length=206

    label_names=["reactivity_{:04d}".format(number+1) for number in range(seq_length)]
    error_label_names=["reactivity_error_{:04d}".format(number+1) for number in range(seq_length)]

    sequences=pl_train.unique(subset=["sequence_id"],maintain_order=True)['sequence'].to_list()
    sequence_ids=pl_train.unique(subset=["sequence_id"],maintain_order=True)['sequence_id'].to_list()
    labels=pl_train[label_names].to_numpy().astype('float16').reshape(-1,2,seq_length).transpose(0,2,1)
    errors=np.zeros_like(labels).astype('float16')
    SN=pl_train['signal_to_noise'].to_numpy().astype('float16').reshape(-1,2)

    SN[:]=10 # set SN to 10 so they don't get masked

    data_dict = {
        'sequences': sequences,
        'sequence_ids': sequence_ids,
        'labels': labels,
        'errors': errors,
        'SN': SN,
    }

    return data_dict

def load_config_from_yaml(file_path):
    """
    Load configuration from a YAML file.
    
    Args:
        file_path (str): Path to YAML configuration file
        
    Returns:
        Config: Configuration object with parameters from YAML file
        
    Raises:
        yaml.YAMLError: If YAML file is malformed
        FileNotFoundError: If file_path doesn't exist
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

def write_config_to_yaml(config, file_path):
    """
    Write configuration to a YAML file.
    
    Args:
        config (Config): Configuration object to save
        file_path (str): Path where to save YAML file
        
    Raises:
        yaml.YAMLError: If configuration cannot be serialized to YAML
        PermissionError: If file_path is not writable
    """
    with open(file_path, 'w') as file:
        yaml.safe_dump(config, file)

def get_distance_mask(L):
    """
    Create a distance-based mask matrix.
    
    Generates a matrix where each element (i,j) is:
    - 1 if i == j (diagonal)
    - 1/distance^2 if i != j, where distance = |i-j|
    
    Args:
        L (int): Size of the square matrix
        
    Returns:
        np.ndarray: L x L matrix with distance-based values
        
    Example:
        >>> mask = get_distance_mask(3)
        >>> print(mask)
        [[1.    0.25  0.11]
         [0.25  1.    0.25]
         [0.11  0.25  1.  ]]
    """
    m=np.zeros((L,L))

    for i in range(L):
        for j in range(L):
            if abs(i-j)>0:
                m[i,j]=1/abs(i-j)**2
            elif i==j:
                m[i,j]=1
    return m

class CSVLogger:
    """
    A simple CSV logger for tracking metrics during training/evaluation.
    
    This class provides methods to create and append to CSV files,
    maintaining consistent column structure.
    
    Attributes:
        columns (list): List of column names for the CSV
        file (str): Path to the CSV file
        
    Example:
        >>> logger = CSVLogger(['epoch', 'loss', 'accuracy'], 'training.csv')
        >>> logger.log([1, 0.5, 0.95])
    """
    
    def __init__(self, columns, file):
        """
        Initialize CSV logger.
        
        Args:
            columns (list): List of column names
            file (str): Path to output CSV file
        """
        self.columns = columns
        self.file = file
        if not self.check_header():
            self._write_header()

    def check_header(self):
        """
        Check if CSV file exists and has headers.
        
        Returns:
            bool: True if file exists, False otherwise
        """
        if path.exists(self.file):
            header=True
        else:
            header=False
        return header

    def _write_header(self):
        """
        Write column headers to CSV file.
        
        Returns:
            CSVLogger: Self for method chaining
        """
        with open(self.file,"a") as f:
            string=""
            for attrib in self.columns:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

    def log(self, row):
        """
        Write a row of data to CSV file.
        
        Args:
            row (list): List of values to write, must match number of columns
            
        Returns:
            CSVLogger: Self for method chaining
            
        Raises:
            Exception: If length of row doesn't match number of columns
        """
        if len(row)!=len(self.columns):
            raise Exception("Mismatch between row vector and number of columns in logger")
        with open(self.file,"a") as f:
            string=""
            for attrib in row:
                string+="{},".format(attrib)
            string=string[:len(string)-1]
            string+="\n"
            f.write(string)
        return self

if __name__=='__main__':
    print(load_config_from_yaml("configs/sequence_only.yaml"))
