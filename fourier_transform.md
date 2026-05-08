# Transformada de fourier

Tenemos señales que viven en z = f(x,y) señales tendran:
- frecuencias horizontales u si la oscilacion ocurre sobre el eje x. Las frecuencias verticales no cambian (cilindro)
- Frecuencias verticales b si la osiclacion ocurre sobre el eje y. Las frecuencias horizontales no cambian (cilindro)
- frecuencias (u,v) si la frecuencia ocurre en ambas direcciones espaciales. 

 Ejemplo de señales horizontales:
![Ejemplo de señales horizontales:](/tex_img/image.png)

Podemos Pensar que estas señales, z representa el nivel de gris de la misma y las coordenadas espaciales son pixeles de una imagen. Y la frecuencia representa cuantas veces suben y baja por el eje de la imagen.

# Espacio base
En el dominio de las frecuencias podemos pensar un espacio bidimensional de frecuencias horizontales y verticales. En ella podemos representar un conjunto de señales sinusoidales bidimensionales que describen una onda sinosidal con sus frecuencias u,v. 

Por ejemplo: Supongamos que tenemos una señal u=0, v=4 representa una señal sinosidal que tiene frecuencia vertical cuatro (cuatro ciclos en el eje y) y frecuencia horizontal cero (no varía a lo largo del renglon o eje x). Su representación en frecuencia se veria como: 
![aaaa](/tex_img/image-1.png)

Por ultimo en esta grilla, puede haber combinacion de frecuencias (u,v), donde se verian las sinusoidales inclinadas. Supongamos que tenmos una señal u=1 y v=4, representa que en el eje vertical varia mas que en el eje horizontal, por lo que es de esperar que la sinusoidal proyeccion con frecuencia y oscile mas que la respectiva en x
![alt text](/tex_img/image3.png).

Otro ejemplo puede ser sumar u=2 v=1, obtenemos como resultado una imagen que horizontalmente tiene frecuencia dos (ver sinusoidal proyectada) y verticalmente tiene frecuencia 1 (solo oscila una vez)
![alt text](/tex_img/image4.png)

Podemos pesar los coeficientes de estas señales. Si usamos como se ve la imagen de 0.5, 0.5, en el dominio frecuencial luciria como dos impulsos y la alutra seria el coeficiente. Al utilizar 0.5 sería un promedio de ambas señales. 

No obstante, yo podría intensificar con otras ponderiaciones las combinaciones de lass señales, cambiando el resultado de las mismas en materia de intensidad (nivel de gris alcanzado, amplitud). Por ejemplo, si nosotros pesamos la frecuencia horizontal de 2 con 0,2 y la frecuencia vertical con 0.8 el resultado es una sinosuidal de frecuencias verticales de mayor amplitud, por lo que alcanza niveles de brillo mas extremos en estos ejes (muy negro y muy blanco) mientras que en el eje horizontal es mas bajo (mas blanco)
![alt text](/tex_img/image5.png)


Estas grilla funciona como una base para el espacio de imagen y se denomina **espacio de imagenes base** y sirven para reconstruir una imagen en sus componentes frecuenciales mediante una combinacion lineal de las componentes. 

# Sobre la transofmrada de fourier

Tenemos la formula $$F(\omega, \nu) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(t, z) e^{-j2\pi(\omega t + \nu z)} dt \, dz$$ en dos dimensiones donde $$e^{-j2\pi(\omega t + \nu z)}$$ representa la base para traducir al dominio frecuencial.


En su version discretizada (DFT Y FFT) tenemos $$F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$ para ir al dominio frecuencial y para volver $$f(x, y) = \frac{1}{MN} \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) e^{j 2\pi \left( \frac{ux}{M} + \frac{vy}{N} \right)}$$

Si yo tengo un F(u,v) alto, implica que en la combinación u, v correspondiente, implica que en esa frecuencia hay mucha informacion presente en la imagen y es por eso que en la reconstruccion, pesa mas esa señal.


Las funciones bases que antes habiamos visto es justamente esa exponencial compleja de la formula de la transformada de fourier ya que sabemos que $e^{j\theta}= cos(\theta) + j sin(\theta)$ y la base de sinusoidales representa

![alt text](/tex_img/image6.png) y tienen una forma radial con radio las frecuencia correspondiente.


La transformada de fourier discreta original  de una imagen luce algo como 
![alt text](/tex_img/image7.png) donde el 0 vertical y 0 horizontal se encuentra sobre el cursos indicado. Por lo general, se desplaza esta imagen y se la centra en 0.0 para obtener ![alt text](/tex_img/image8.png) donde ahora el centro de la transformada es el (0,0), extendiendose a valores negativos aprovechando la periodicidad de la operacion de la transformada de fourier. Esta imagen tiene mucha informacion en las frecuencias bajas y poca informacion en las frecuencias altas.

Podriamos aplicar filtros (por ejemplo un filtro pasabajo, gaussiana) sobre la transforamda de fourier (tambien llamada espectro)  para limpiar las frecuencias altas (detalles de la imagen) y volver al dominio original de la imagen.