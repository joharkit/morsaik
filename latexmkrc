# latexmk configuration file
# See: http://mirrors.ctan.org/support/latexmk/latexmk.pdf

## main tex file of the project
@default_files = ('manual/morsaik_manual.tex');

## build directory, to keep source directory clean
$out_dir = "build";

## enable pdf output
$pdf_mode = 1;        # tex -> pdf

## extra files to be cleaned on 'latexmk -c'
$clean_ext = "pdata bbl run.xml %R-blx.bib bbl nav out snm acn acr alg glg glo gls ist";


# optional: enable to use tikzexternalize

## fix to add extra folder for tikz tikzexternalize
## Source: https://tex.stackexchange.com/questions/243935/error-using-tikz-externalize-cant-write-md5-file/360284#360284
## Source: https://tex.stackexchange.com/questions/206695/latexmk-outdir-with-include#comment756748_206986
#system ("mkdir -p $out_dir/$out_dir");

## allow shell escape
#$pdflatex = 'pdflatex --shell-escape %O %S';

## allow for compiling glossaries
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');
sub run_makeglossaries {
  if ( $silent ) {
    system "makeglossaries -q '$_[0]'";
  }
  else {
    system "makeglossaries '$_[0]'";
  };
}
