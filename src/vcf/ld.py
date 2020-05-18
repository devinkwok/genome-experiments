import vcf

reader = vcf.Reader(open('data/sample.vcf'))

for r in reader:
    print(r.genotype('SAMP001').data)