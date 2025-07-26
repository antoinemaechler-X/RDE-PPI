import pickle
with open('data/SKEMPI_v2_cache.pkl', 'rb') as f:
    cache = pickle.load(f)
print('1dan' in cache)  # Should print True if present