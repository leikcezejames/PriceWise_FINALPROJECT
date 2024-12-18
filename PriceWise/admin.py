from django.contrib import admin
from .models import Product, ProductPrice, Prediction

admin.site.register(Product)
admin.site.register(ProductPrice)
admin.site.register(Prediction)