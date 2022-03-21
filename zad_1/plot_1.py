import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Dane na osi Y losowe, więc czasami (dla niektórych danych) tekst zlewa się z wykresem,
# i ogólnie losowe dane źle wpływają na czytelność i estetykę, z oryginalnymi wyglądałoby lepiej,
# z tego powodu też postanowiłem dać tytuł nad wykresem, a nie w jego rogu jak w NYT
X = np.linspace(start=1880, stop=2018, num=139)
Y = np.random.random(139)
mean = np.mean(Y[:20])
Y = Y - mean


fig, ax = plt.subplots()

ax.plot(X, Y, color='grey')
ax.scatter(X, Y, s=200, c=Y, cmap='coolwarm', marker='.', edgecolors='grey')

ax.set_xticks(np.arange(1880, 2011, 10))
ax.set_yticks(np.array([0.0]))

ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.1f"))

# linie co 0.2 na osi y
ax.grid(axis='y', which='minor', linestyle=':')
ax.grid(axis='y', which='major', linestyle='-')

# niewidzialne osie
for direction in ('top', 'right', 'bottom', 'left'):
    ax.spines[direction].set_visible(False)

# Opisy konkretnych markerów
ax.annotate(1904, (X[1904 - 1880], Y[1904 - 1880]))
ax.annotate(1944, (X[1944 - 1880], Y[1944 - 1880]))
ax.annotate(1998, (X[1998 - 1880], Y[1998 - 1880]))
ax.annotate(2016, (X[2016 - 1880], Y[2016 - 1880]))
ax.annotate(2018, (X[2018 - 1880], Y[2018 - 1880]), weight='bold')

# Tekst
ax.text(2015, 0.02, 'HOTTER THAN THE\n1880-1899 AVERAGE', size='large')
ax.text(2015, -0.04, 'COLDER', size='large')
mid = (fig.subplotpars.left + fig.subplotpars.right) / 2  # bez tego suptitle i title mają delikatny offset w osi x
plt.suptitle('Rising Global Temperature', size='xx-large', weight='bold', x=mid)
plt.title('How much cooler or warmer every year\nwas compared with the average\ntemperature of the late 19th century', size='large')


fig.show()
