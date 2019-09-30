latex logo.tex
dvips logo.dvi
ps2pdf -dPDFSETTINGS=/prepress -dEmbedAllFonts=true logo.ps
gs -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -dEmbedAllFonts=true -sOutputFile=logo.pdf -f logo.ps
rm logo.dvi logo.ps logo.aux logo.log
convert -density 1000 -depth 10 -quality 95 logo.pdf logo.png
