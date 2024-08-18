from statistics import mean
from scipy.stats import sem

if __name__ == "__main__":
    ours = {'BookOrder': [1.0, 0.96, 0.98, 0.98, 0.92], 'Club': [0.34, 0.32, 0.28, 0.28, 0.42],
            'Election': [0.58, 0.56, 0.6, 0.46, 0.7], 'Series': [0.2, 0.14, 0.14, 0.22, 0.24],
            'Verdict': [0.3, 0.42, 0.24, 0.44, 0.26]}
    facloc = {'BookOrder': [0.4411764705882353, 0.4492753623188406, 0.5, 0.463768115942029, 0.45588235294117646],
              'Club': [0.2987012987012987, 0.21568627450980393, 0.2537313432835821, 0.27102803738317754,
                       0.22429906542056074],
              'Election': [0.49019607843137253, 0.4117647058823529, 0.35294117647058826, 0.54, 0.4339622641509434],
              'Series': [0.15789473684210525, 0.13559322033898305, 0.18181818181818182, 0.15873015873015872,
                         0.2028985507246377],
              'Verdict': [0.16363636363636364, 0.10526315789473684, 0.08620689655172414, 0.1, 0.08928571428571429]}
    kmeans = {'BookOrder': [0.0, 0.0, 0.0, 0.0, 0.0], 'Club': [0.0, 0.0, 0.0, 0.0, 0.0],
              'Election': [0.0, 0.0, 0.0, 0.0, 0.0],
              'Series': [0.0, 0.0, 0.0, 0.0, 0.0], 'Verdict': [0.0, 0.0, 0.0, 0.0, 0.0]}
    pca = {'BookOrder': [0.92, 0.94, 0.96, 0.9, 0.9], 'Club': [0.24, 0.28, 0.32, 0.26, 0.32],
           'Election': [0.2, 0.2, 0.22, 0.16, 0.2], 'Series': [0.32, 0.26, 0.26, 0.24, 0.22],
           'Verdict': [0.18, 0.2, 0.18, 0.2, 0.16]}
    em = {'BookOrder': [0.34, 0.52, 0.46, 0.42, 0.32], 'Club': [0.2, 0.16, 0.24, 0.12, 0.2],
          'Election': [0.32, 0.26, 0.36, 0.34, 0.28], 'Series': [0.3, 0.3, 0.2, 0.34, 0.24],
          'Verdict': [0.26, 0.3, 0.24, 0.26, 0.32]}

    filenames = ['Club', 'Election', 'Series', 'Verdict', 'BookOrder']

    print("ours: ")
    for filename in filenames:
        print("file: %s, mean: %s, std: %s" % (filename, mean(ours[filename]), sem(ours[filename])))

    print("----------------------------------------------------")

    print("em: ")
    for filename in filenames:
        print("file: %s, mean: %s, std: %s" % (filename, mean(em[filename]), sem(em[filename])))

    print("facloc: ")
    for filename in filenames:
        print("file: %s, mean: %s, std: %s" % (filename, mean(facloc[filename]), sem(facloc[filename])))

    print("kmeans: ")
    for filename in filenames:
        print("file: %s, mean: %s, std: %s" % (filename, mean(kmeans[filename]), sem(kmeans[filename])))

    print("pca: ")
    for filename in filenames:
        print("file: %s, mean: %s, std: %s" % (filename, mean(pca[filename]), sem(pca[filename])))
