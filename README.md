# biblio-checker

Verificador automático de referencias bibliográficas para artículos científicos.

Lee un artículo en Markdown, extrae la sección de referencias y verifica cada una contra [CrossRef](https://www.crossref.org/) y [Semantic Scholar](https://www.semanticscholar.org/).

## Requisitos

- Python 3.9+
- Conexión a Internet
- Sin dependencias externas (solo bibliotecas estándar de Python)

## Uso rápido

```bash
python3 biblio_checker_direct.py \
  --pub-id MiArticulo \
  --draft mi_paper.md \
  --output-dir ./resultados
```

## Resultado

Genera `REFERENCES_VERIFIED.md` con una tabla por referencia:

| Estado | Significado |
|--------|-------------|
| **VERIFIED** | Referencia verificada — título coincide con la API |
| **MISMATCH** | Paper encontrado pero el título difiere |
| **NOT_FOUND** | No encontrado en CrossRef ni Semantic Scholar |
| **UNVERIFIABLE** | Libro (ISBN) u otro recurso no verificable por API |

## Parámetros

| Parámetro | Obligatorio | Descripción |
|-----------|-------------|-------------|
| `--pub-id` | Sí | Identificador del artículo (aparece en el informe) |
| `--draft` | Sí | Ruta al archivo Markdown |
| `--output-dir` | No | Carpeta de salida (default: misma que el draft) |
| `--json` | No | Salida JSON por consola |
| `--debug` | No | Muestra errores HTTP detallados |

## Formatos soportados

La sección de referencias debe llamarse `## References`, `## Referencias`, `## Bibliography` o `## Bibliografía`.

Referencias numeradas en formato:
- `1. Autor (año). Título...`
- `[1] Autor (año). Título...`

## API key de Semantic Scholar (opcional)

Sin API key funciona, pero con límite de ~20 peticiones/minuto. Con key gratuita: 1 petición/segundo.

Para configurar: crea `~/.env.api_keys` con:
```
SEMANTIC_SCHOLAR_API_KEY=tu_clave
```

Solicitar key gratuita: https://www.semanticscholar.org/product/api#api-key-form

## Documentación

Ver `biblio_checker_manual.docx` para instrucciones detalladas con ejemplos y solución de problemas.

## Licencia

MIT

---

Desarrollado en [PLOCAN](https://www.plocan.eu) — Plataforma Oceánica de Canarias.
