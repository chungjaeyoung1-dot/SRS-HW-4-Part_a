import kaggle

kaggle.api.authenticate()

kaggle.api.dataset_download_files('gabrielkahen/music-listening-data-500k-users', path='.', unzip=True)