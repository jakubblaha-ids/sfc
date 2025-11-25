ZIP_NAME = 9-xblaha36.zip
FILES_TO_PACK = doc/doc.pdf doc/doc.tex mhn-localization requirements.txt maps install-and-run.sh

.PHONY: pack clean

pack:
	zip -r $(ZIP_NAME) $(FILES_TO_PACK) -x "*.pyc" -x "*__pycache__*" -x "*.DS_Store"

clean:
	rm -f $(ZIP_NAME)
