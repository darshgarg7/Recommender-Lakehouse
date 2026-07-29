locals {
  schemas = toset(["landing", "bronze", "silver", "gold", "features", "ml", "serving", "monitoring"])
  catalog_schema_pairs = {
    for pair in setproduct(var.catalog_names, local.schemas) : "${pair[0]}.${pair[1]}" => {
      catalog = pair[0]
      schema  = pair[1]
    }
  }
}
resource "databricks_catalog" "marketplace" {
  for_each     = var.catalog_names
  name         = each.value
  storage_root = var.storage_root_by_catalog[each.value]
  comment      = "Marketplace recommender ${each.value} environment"
}

resource "databricks_schema" "marketplace" {
  for_each     = local.catalog_schema_pairs
  catalog_name = databricks_catalog.marketplace[each.value.catalog].name
  name         = each.value.schema
  comment      = "Managed by Terraform; do not create ad hoc production tables"
}
