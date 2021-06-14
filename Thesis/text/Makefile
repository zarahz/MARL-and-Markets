# --------------------------------------------------------
#      MDSG Latex Framework
#      =========================================
#      File:                  Makefile
#      Author(s):             Michael Duerr
#      Version:               1
#      Creation Date:         30. Mai 2010
#      Creation Date:         30. Mai 2010
#
# --------------------------------------------------------
TARGET=main

all:
	pdflatex $(TARGET).tex
	bibtex $(TARGET).aux
	pdflatex $(TARGET).tex
	pdflatex $(TARGET).tex

clean:
	-find . \( -name "*.aux" -o -name "*.lof" \
                -o -name "*.log" -o -name "*.out" \
		-o -name "*.toc" -o -name "*.bbl" \
		-o -name "*.nls" -o -name "*.blg" \
                -o -name "*.lot" -o -name "*.lol" \) | xargs rm
	-rm $(TARGET).pdf
