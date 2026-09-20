# Hardened update для `scarlettnik/area`

Этот пакет накладывается поверх текущего `main`. Он закрывает найденный на реальном запуске failure mode: solver не должен завершать весь расчет аварией, если один refined finalist расходится с независимым validator после сериализации.

Что изменено:

- `optimize_network.py`: round-trip validation каждого кандидата; invalid refinement откатывается на original finalist; один invalid finalist больше не валит весь запуск; в export попадают только независимо валидные варианты; финальные файлы валидируются повторно после выставления rank.
- `optimize_network.py`: небольшой search-only safety margin (по умолчанию 1 см) для линейных коммуникаций; exact validator остается неизменным.
- `routing_alns.py`: coverage-first acceptance закреплен и для simulated annealing; добавлены `retie` и `hotspot` destroy operators для смены места присоединения/перестройки дорогих ветвлений.
- `routing_quality.py`: UTM-диагностика лишних поворотов, micro-bends, backtracking и detour ratio; не подменяет официальный score.
- `run_portfolio.py`: детерминированный multi-seed ALNS portfolio и выбор по `max coverage -> min official score`.
- `REQUIREMENTS_TRACEABILITY.md`: чек-лист актуального техприложения и разъяснений.
- `.gitignore`: убраны IDE/cache/results из репозитория.

## Как применить

Распаковать файлы поверх актуального `main` репозитория, затем выполнить `uv sync` и `uv run python -m unittest -v`.

Одиночный production-run:

uv run python optimize_network.py --input "Датасет скорректированный.geojson" --output-dir routing_results_final --mode 2d --solver alns --search-seconds 600 --refine-seconds 120 --beam-width 8 --routes 5 --neighbors 3 --iterations 5000 --seed 17

Для устойчивого результата лучше portfolio:

uv run python run_portfolio.py --input "Датасет скорректированный.geojson" --output-dir routing_results_portfolio --mode 2d --seeds 17,23,41,73,101 --search-seconds 300 --refine-seconds 90 --iterations 2500 --beam-width 6 --routes 4 --neighbors 3

Финал: `routing_results_portfolio/best/best_variant.geojson`.

Важно: это best-found эвристическое решение. Без точного глобального solver/certificate утверждать математически доказанный глобальный минимум score нельзя.
