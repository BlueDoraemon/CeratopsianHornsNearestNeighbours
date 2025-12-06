# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Load Data
data = {
    'Specimen#': ['CMN 8535', 'ROM 802', 'TMP 1983.001.0001', 'UW 2419', 'ROM 769', 'ROM 1439', 'TMM 43093-1', 'AMNH 5402', 'CMN 1254', 'CMN 2280', 'CMN 8800', 'CMN 8801', 'ROM 839', 'ROM 843', 'TMP 1981.19.175', 'TMP 2014.004.0001', 'UALVP 40', 'YPM 2016', 'TMP 2002.57.007', 'AMNH 5401', 'TMP 1979.011.0147', 'TMP 2005.055.0001', 'OMNH 10165', 'MOR 1110', 'MOR 1120', 'MOR 1604', 'MOR 2551', 'MOR 2574', 'MOR 2923', 'MOR 2952', 'MOR 2982', 'MOR 3010', 'MOR 3045', 'MOR 3081', 'USNM 4928', 'ANSP 15192', 'MOR 981', 'MOR 1122', 'TMP 1985.10.13', 'CMN 41357', 'TMP 1987.45.1', 'AMNH 5239', 'AMNH 5351', 'CMN 347-Laval', 'CMN 348', 'CMN 1173', 'CMN 11837', 'CMN 8795', 'CMN 8798', 'ROM 767', 'TMP 1992.82.1', 'TMP 1997.85.1', 'UALVP 11735', 'YPM 2015', 'AMNH 3999', 'CMN 8790', 'CMN 9485', 'Drumheller', 'TMP 1986.55.258', 'TMP 1987.55.156', 'TMP 1989.55.188', 'TMP 1989.55.1234', 'DMNH 22558', 'TMP 2002.76.1', 'MOR 492', 'NHM R 8648', 'CMN 344', 'TMP 2009.90.01', 'TMP 2005.12.58'],
    'Taxon': ['Anchiceratops ornatus', 'Anchiceratops ornatus', 'Anchiceratops ornatus', 'Anchiceratops ornatus', 'Arrhinoceratops brachyops', 'Arrhinoceratops brachyops', 'Agujaceratops mariscalnsis', 'Chasmosaurus belli', 'Chasmosaurus belli', 'Chasmosaurus russelli', 'Chasmosaurus russelli', 'Chasmosaurus sp.', 'Chasmosaurus belli', 'Chasmosaurus belli', 'Chasmosaurus cf. C. russelli', 'Chasmosaurus sp.', 'Chasmosaurus sp.', 'Chasmosaurus belli', 'Eotriceratops xerinsularis', 'Mojoceratops perifania', 'Mojoceratops perifania', 'Regaliceratops peterhewsi', 'Titanoceratops ouranos', 'Triceratops prorsus', 'Triceratops horridus', 'Triceratops sp.', 'Triceratops sp.', 'Triceratops prorsus', 'Triceratops prorsus', 'Triceratops sp.', 'Triceratops sp.', 'Triceratops sp.', 'Triceratops sp.', 'Triceratops horridus', 'Triceratops sp.', 'Torosaurus latus', 'Torosaurus sp.', 'Torosaurus sp.', 'Torosaurus sp.', 'Vagaceratops irvinensis', 'Vagaceratops irvinensis', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Centrosaurus apertus', 'Monoclonius recurvicornus', 'Monoclonius lowei', 'Pachyrhinosaurus canadensis', 'Pachyrhinosaurus canadensis', 'Pachyrhinosaurus lakustai', 'Pachyrhinosaurus lakustai', 'Pachyrhinosaurus lakustai', 'Pachyrhinosaurus lakustai', 'Pachyrhinosaurus perotorum', 'DPP Pachyrhinosaur', 'Rubeosaurus ovatus', 'Spinops sternbergorum', 'Styracosaurus abertensis', 'Styracosaurus abertensis', 'Styracosaurus abertensis'],
    'Nasal Horncore Length (mm)': [97, 82.5, 161.5, 128.5, 103, 108, 150, 144, 150, 133, 123, 196, 236, 202, 138, 182, 135, 115, 148, 130, 118, 257, 160, 83, 58, 265, 220, 210, 290, 270, 100, 65, 160, 139, 110, 106, 128, 145, 98, 115, 100, 358, 458, 314, 204, 320, 218, 200, 237, 293, 206, 221, 208, 318, 175, 198, 235, 278, 165, 175, 150, 210, 321, 164, 390, 188, 534, 167, 234],
    'Postorbital Horncore Length (mm)': [315, 620, 600, 672.5, 414, 457, 430, 67, 216, 133, 173, 73, 103, 64, 130, 108, 172, 47.5, 717, 370, 306, 147, 910, 500, 490, 605, 500, 510, 550, 600, 500, 470, 583, 500, 780, 380, 853, 434, 357, 42, 43, 88.5, 53.5, 50.5, 154, 51, 111, 78, 58, 96.5, 34, 75.5, 145, 69.5, 166, 45.5, 95, 100, 65, 80, 70, 75, 72, 79, 150, 77, 62.5, 53.5, 43]
}
df = pd.DataFrame(data)

# Mapping Function for Groups
def categorise_taxon(taxon_name):
    genus = taxon_name.split()[0]
    
    triceratopsini = ['Triceratops', 'Torosaurus', 'Eotriceratops', 'Titanoceratops', 'Regaliceratops']
    centrosaurinae = ['Centrosaurus', 'Styracosaurus', 'Pachyrhinosaurus', 'Monoclonius', 'Rubeosaurus', 'Spinops', 'DPP'] # DPP is the Pachyrhinosaur
    
    # Default to Chasmosaurinae (Early/Basal) if not in the other two specific lists
    if genus in triceratopsini:
        return 'Triceratopsini'
    elif genus in centrosaurinae:
        return 'Centrosaurinae'
    else:
        return 'Early Chasmosaurinae'

df['Group'] = df['Taxon'].apply(categorise_taxon)

# Prepare Data for Modeling
X = df[['Nasal Horncore Length (mm)', 'Postorbital Horncore Length (mm)']].values
y = df['Group'].values

# Encode Labels (3 Groups)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model (KNN)
knn = KNeighborsClassifier(n_neighbors=10, weights='distance') # Increased neighbors for smoother 3-class boundaries
knn.fit(X_scaled, y_encoded)

# Confidence Ellipse Function
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Plotting
fig, ax = plt.subplots(figsize=(14, 10))


# Meshgrid for Sigmoid/Gradient background
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))


# Predict for background (Use Max Probability for intensity)
Z_proba = knn.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z_max = np.max(Z_proba, axis=1).reshape(xx.shape)
Z_class = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Create a custom colormap overlay
# We map the class ID to a base color, then scale alpha by Z_max (certainty)
from matplotlib.colors import ListedColormap
# Define base colors for the 3 groups: Triceratopsini (Blue), Early Chas (Green), Centrosaurinae (Red)
cmap_light = ListedColormap(['#FFCCCC', '#CCFFCC', '#CCCCFF'])
ax.contourf(xx, yy, Z_class, cmap=cmap_light, alpha=0.4) # Base regions
# Overlay intensity contour to give "sigmoid center" feel
ax.contourf(xx, yy, Z_max, cmap='Greys', alpha=0.2, levels=10)
# Meshgrid for Background
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Scatter and Rings
colors = ['red', 'green', 'blue'] # Matches alphabetical order of groups usually
markers = ['o', 's', '^']

for idx, group_name in enumerate(le.classes_):
    # Filter points for this group
    group_mask = (y_encoded == idx)
    points = X_scaled[group_mask]
    original_labels = df.loc[group_mask, 'Taxon'].values
    
    # Plot Group Points
    ax.scatter(points[:, 0], points[:, 1], 
               label=group_name, c=colors[idx], marker=markers[idx], edgecolor='k', s=80, zorder=10)
    
    # Plot Rings (Standard Deviations)
    confidence_ellipse(points[:, 0], points[:, 1], ax, n_std=1, edgecolor=colors[idx], linestyle='-', alpha=0.9, linewidth=2)
    confidence_ellipse(points[:, 0], points[:, 1], ax, n_std=2, edgecolor=colors[idx], linestyle='--', alpha=0.6)
    
    # Add Taxon Labels
    for i, txt in enumerate(original_labels):
        # Shorten label to Genus only to prevent massive clutter, or keep full if strictly requested
        # Using full taxon name but small font
        ax.text(points[i, 0]+0.02, points[i, 1]+0.02, txt, fontsize=8, alpha=0.7)

ax.set_title("Ceratopsian Horn Morphology: 3-Group Classification\n(Decision Boundaries + SD Rings)", fontsize=14)
ax.set_xlabel("Standardised Nasal Horn Length", fontsize=12)
ax.set_ylabel("Standardised Postorbital Horn Length", fontsize=12)
ax.legend(loc='upper right', title="Clade/Group")
plt.grid(False)
plt.show()


# %%
