import datetime
from typing import Optional, Dict, Any, Callable, List, Tuple 
from RAG_BOT.src.logger import logger


class FilterProcessor:
    """Handles filter preparation and date validation."""
    
    @staticmethod
    def prepare_filters(date_filter: Optional[str], language: Optional[str]) -> Tuple[Dict[str, Any], Optional[str]]:
        """Prepare search filters and return base kwargs with active date."""
        conditions = []
        active_date = None
        
        if date_filter:
            active_date = FilterProcessor._process_date_filter(date_filter, conditions)
        
        if language:
            conditions.append({"language": language.lower()})
            logger.info(f"Language filter condition prepared: {language.lower()}")
        
        search_kwargs = {}
        if conditions:
            search_kwargs["filter"] = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            logger.info(f"Base filter for Chroma: {search_kwargs.get('filter')}")
        
        return search_kwargs, active_date
    
    @staticmethod
    def _process_date_filter(date_filter: str, conditions: List[Dict]) -> Optional[str]:
        """Process date filter and add to conditions if valid."""
        try:
            filter_date_obj = datetime.datetime.strptime(date_filter, '%Y-%m-%d')
            active_date = filter_date_obj.strftime('%Y-%m-%d')
            date_condition = {"date": active_date}
            conditions.append(date_condition)
            logger.info(f"Date filter condition prepared: {date_condition}")
            return active_date
        except ValueError:
            logger.warning(f"Invalid date format '{date_filter}'. Date filter will not be used.")
            return None