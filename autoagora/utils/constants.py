# Default agora variables are set here
DEFAULT_AGORA_VARIABLES = {"DEFAULT_COST": 50}

AGORA_DEFAULT_COST_MODEL = "default => $DEFAULT_COST * $GLOBAL_COST_MULTIPLIER;"

MU = 0.4
SIGMA = 0.2

AGORA_ENTRY_TEMPLATE = """\
# Generated by AutoAgora {{aa_version}}

{% if manual_entry is not none %}
{{manual_entry}}
{% endif %}
{% for frequent_query in most_frequent_queries %}

# count:        {{frequent_query.count}}
# min time:     {{frequent_query.min_time}}
# max time:     {{frequent_query.max_time}}
# avg time:     {{frequent_query.avg_time}}
# stddev time:  {{frequent_query.stddev_time}}
{{frequent_query.query}} => {{frequent_query.avg_time}} * $GLOBAL_COST_MULTIPLIER;
{% endfor %}
default => $DEFAULT_COST * $GLOBAL_COST_MULTIPLIER;\
"""
GET_MFQ_QUERY_LOGS = """\
SELECT
    query,
    count_id,
    min_time,
    max_time,
    avg_time,
    stddev_time
FROM
    query_skeletons
INNER JOIN
(
    SELECT
        query_hash as qhash,
        count(id) as count_id,
        Min(query_time_ms) as min_time,
        Max(query_time_ms) as max_time,
        Avg(query_time_ms) as avg_time,
        Stddev(query_time_ms) as stddev_time 
    FROM
        query_logs
    WHERE
        subgraph = $1
        AND query_time_ms IS NOT NULL
    GROUP BY
        qhash
    HAVING
        Count(id) >= $2
) as query_logs
ON
    qhash = hash
ORDER BY
    count_id DESC
"""

GET_MFQ_MRQ_LOGS = """\
SELECT
    query,
    count_id,
    min_time,
    max_time,
    avg_time,
    stddev_time
FROM
    query_skeletons
INNER JOIN
(
    SELECT
        query_hash as qhash,
        count(id) as count_id,
        Min(query_time_ms) as min_time,
        Max(query_time_ms) as max_time,
        Avg(query_time_ms) as avg_time,
        Stddev(query_time_ms) as stddev_time 
    FROM
        mrq_query_logs
    WHERE
        subgraph = $1
        AND query_time_ms IS NOT NULL
    GROUP BY
        qhash
    HAVING
        Count(id) >= $2
) as query_logs
ON
    qhash = hash
ORDER BY
    count_id DESC
"""
