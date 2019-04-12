all: bksyn_loader bksyn_model

bksyn_loader: bks_dataset.py dataloader.py icdar_dataloader.py utils.py
	zip bks/bks.data.zip bks_dataset.py dataloader.py icdar_dataloader.py utils.py

bksyn_model: bks_model.py model.py utils.py losses.py anchors.py csv_eval.py
	zip bks/bks.model.zip bks_model.py model.py utils.py losses.py anchors.py csv_eval.py
